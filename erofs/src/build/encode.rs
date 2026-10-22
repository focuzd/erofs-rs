use alloc::{collections::BTreeMap, format, string::String, vec::Vec};
use bytes::{Buf, BufMut};
use crc32c::{crc32c, crc32c_append};
use std::io;

use super::{Compression, InodeFormat, Metadata, Options, SpecialFile, State};
use crate::{
    Error, Result, Xattrs,
    types::{
        Dirent, DirentFileType, FileMode, InodeCompact, InodeExtended, Layout, MAGIC_NUMBER,
        SUPER_BLOCK_OFFSET, SuperBlock, XattrEntry, XattrHeader,
    },
};

pub(super) const BLOCK_SIZE: usize = 4096;
pub(super) const INODE_SLOT_SIZE: usize = 32;
pub(super) const LZMA_DICT_SIZE: u32 = 64 * 1024;
pub(super) const ZSTD_WINDOW_LOG: u8 = 17;
pub(super) const ZERO_BLOCK: [u8; BLOCK_SIZE] = [0; BLOCK_SIZE];
pub(super) const ROOT_OFFSET: usize = SUPER_BLOCK_OFFSET as usize + SuperBlock::size();
pub(super) const MAX_XATTR_SIZE: usize = size_of::<XattrHeader>() + (u16::MAX as usize - 1) * 4;

const ADVISE_COMPACTED_2B: u16 = 1;
const ADVISE_BIG_PCLUSTER: u16 = 2 | 4; // Both HEAD types must agree for Compact indexes.
const ADVISE_INLINE_TAIL: u16 = 8;
const ADVISE_FRAGMENT: u16 = 0x20;

const FEATURE_ZERO_PADDING: u32 = 1;
const FEATURE_COMPRESSION_CONFIG: u32 = 2;
const FEATURE_CHUNKED_FILE: u32 = 4;
const FEATURE_INLINE_TAILS: u32 = 0x10;
const FEATURE_FRAGMENTS: u32 = 0x20;
const FEATURE_XATTR_PREFIXES: u32 = 0x40;
const FEATURE_48BIT: u32 = 0x80;

pub(super) fn xattr_name(name: &[u8]) -> Result<(u8, &[u8])> {
    if name.is_empty() || name.len() > 255 || name.contains(&0) {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "invalid xattr name").into());
    }
    for index in [1, 2, 3, 4, 6] {
        if let Some(suffix) = name.strip_prefix(crate::xattr::namespace(index)?) {
            if suffix.is_empty() != matches!(index, 2 | 3) {
                return Err(
                    io::Error::new(io::ErrorKind::InvalidInput, "invalid xattr name").into(),
                );
            }
            return Ok((index, suffix));
        }
    }
    Err(Error::NotSupported(format!(
        "xattr namespace for {:?}",
        String::from_utf8_lossy(name),
    )))
}

pub(super) fn xattr_entry_size(name_len: usize, value_len: usize) -> Result<usize> {
    if value_len > u16::MAX as usize {
        return Err(Error::Overflow("xattr value size"));
    }
    Ok((size_of::<XattrEntry>() + name_len + value_len).next_multiple_of(4))
}

fn xattrs(attrs: &Xattrs) -> Result<Vec<u8>> {
    if attrs.is_empty() {
        return Ok(Vec::new());
    }
    let mut data = vec![0; size_of::<XattrHeader>()];
    for (name, value) in attrs {
        let (index, suffix) = xattr_name(name)?;
        let size = data.len() + xattr_entry_size(suffix.len(), value.len())?;
        if size > MAX_XATTR_SIZE {
            return Err(Error::Overflow("inode xattr size"));
        }
        data.put_u8(suffix.len() as u8);
        data.put_u8(index);
        data.put_u16_le(value.len() as u16);
        data.put_slice(suffix);
        data.put_slice(value);
        data.resize(size, 0);
    }
    Ok(data)
}

#[derive(Default)]
pub(super) struct Attributes {
    records: BTreeMap<Vec<u8>, u32>,
    bytes: usize,
    pub prefixes: Vec<Vec<u8>>,
    pub prefix_start: u32,
    table_block: u32,
    pub table: Vec<u8>,
}

impl Attributes {
    pub fn rewrite(&mut self, node: &mut Node, options: &Options) {
        if node.xattrs.is_empty() || (!options.shared_xattrs && !options.xattr_prefixes) {
            return;
        }
        let mut shared = Vec::new();
        let mut inline = Vec::new();
        let mut entries = &node.xattrs[size_of::<XattrHeader>()..];
        while !entries.is_empty() {
            let name_len = usize::from(entries[0]);
            let value_len = usize::from((&entries[2..]).get_u16_le());
            let len = (4 + name_len + value_len).next_multiple_of(4);
            let mut record = entries[..len].to_vec();
            let name = &entries[4..4 + name_len];
            if options.xattr_prefixes && !matches!(entries[1], 0 | 2 | 3) {
                let id = self
                    .prefixes
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p[0] == entries[1] && name.starts_with(&p[1..]))
                    .max_by_key(|(_, p)| p.len())
                    .map(|(id, _)| id)
                    .or_else(|| {
                        if self.prefixes.len() == 128 {
                            return None;
                        }
                        // ponytail: automatic selection uses the last dot/full suffix;
                        // a global prescan can choose better prefixes if needed.
                        let len = name
                            .iter()
                            .rposition(|&b| b == b'.')
                            .map_or(name.len(), |i| i + 1);
                        if len < 8 {
                            return None;
                        }
                        self.prefixes.push([&entries[1..2], &name[..len]].concat());
                        Some(self.prefixes.len() - 1)
                    });
                if let Some(id) = id {
                    let prefix_len = self.prefixes[id].len() - 1;
                    record.clear();
                    record.put_u8((name_len - prefix_len) as u8);
                    record.put_u8(0x80 | id as u8);
                    record.put_u16_le(value_len as u16);
                    record.extend_from_slice(&entries[4 + prefix_len..4 + name_len + value_len]);
                    record.resize(record.len().next_multiple_of(4), 0);
                }
            }
            if options.shared_xattrs
                && shared.len() < u8::MAX as usize
                && let Some(&id) = self.records.get(&record)
            {
                shared.push(id);
            } else {
                inline.extend_from_slice(&record);
            }
            entries = &entries[len..];
        }
        node.xattrs = vec![0; size_of::<XattrHeader>()];
        node.xattrs[4] = shared.len() as u8;
        for id in shared {
            node.xattrs.put_u32_le(id);
        }
        node.xattrs.extend_from_slice(&inline);
        node.xattr_size = node.xattrs.len();
    }

    pub fn remember(&mut self, node: &Node, options: &Options) {
        if !options.shared_xattrs || node.xattrs.is_empty() {
            return;
        }
        let start = size_of::<XattrHeader>() + usize::from(node.xattrs[4]) * 4;
        let mut offset = node.inode_offset() + node.inode_size() as u64 + start as u64;
        let mut entries = &node.xattrs[start..];
        while !entries.is_empty() {
            let len = (4 + usize::from(entries[0]) + usize::from((&entries[2..]).get_u16_le()))
                .next_multiple_of(4);
            // Shared records have exactly the inline entry encoding. Reference the
            // original stored entry, rather than writing a second attribute table.
            // ponytail: exact cache capped at 256 KiB / 4096 entries and 32-bit IDs;
            // unmatched records stay inline. Add an external index only if needed.
            if len > 4
                && self.bytes + len <= 256 * 1024
                && self.records.len() < 4096
                && let Ok(id) = u32::try_from(offset / 4)
                && !self.records.contains_key(&entries[..len])
            {
                self.records.insert(entries[..len].to_vec(), id);
                self.bytes += len;
            }
            entries = &entries[len..];
            offset += len as u64;
        }
    }

    pub fn allocate_prefixes(&mut self, next_block: &mut u32, packed_size: u64) -> Result<()> {
        if self.prefixes.is_empty() {
            return Ok(());
        }
        self.table_block = *next_block;
        // Older kernels always read prefixes through the packed inode when it
        // exists. Append whole table pages to its logical chunk map, not its buffer.
        let offset = if packed_size != 0 {
            packed_size.next_multiple_of(BLOCK_SIZE as u64)
        } else {
            u64::from(*next_block) * BLOCK_SIZE as u64
        };
        // ponytail: without a packed inode, the table must start below 16 GiB.
        self.prefix_start =
            u32::try_from(offset / 4).map_err(|_| Error::Overflow("xattr prefix address"))?;
        for prefix in &self.prefixes {
            self.table.put_u16_le(prefix.len() as u16);
            self.table.extend_from_slice(prefix);
            self.table.resize(self.table.len().next_multiple_of(4), 0);
        }
        *next_block = next_block
            .checked_add(self.table.len().div_ceil(BLOCK_SIZE) as u32)
            .ok_or(Error::Overflow("xattr prefix blocks"))?;
        Ok(())
    }

    pub fn blocks(&self) -> impl Iterator<Item = MetadataBlock> {
        self.table.chunks(BLOCK_SIZE).enumerate().map(|(i, bytes)| {
            let mut block = ZERO_BLOCK;
            block[..bytes.len()].copy_from_slice(bytes);
            MetadataBlock {
                block: self.table_block + i as u32,
                start: 0,
                data: block,
            }
        })
    }
}

pub(super) struct MetadataBlock {
    pub block: u32,
    // Reopened pages contain only this suffix; preserve the on-disk prefix.
    start: usize,
    pub data: [u8; BLOCK_SIZE],
}

impl MetadataBlock {
    pub fn new(block: u32) -> Self {
        Self::from_offset(u64::from(block) * BLOCK_SIZE as u64)
    }

    pub fn from_offset(offset: u64) -> Self {
        Self {
            block: (offset / BLOCK_SIZE as u64) as u32,
            start: (offset % BLOCK_SIZE as u64) as usize,
            data: ZERO_BLOCK,
        }
    }

    pub fn offset(&self) -> u64 {
        u64::from(self.block) * BLOCK_SIZE as u64 + self.start as u64
    }

    pub fn bytes(&self) -> &[u8] {
        &self.data[self.start..]
    }

    pub fn inline_data_mut(&mut self, node: &Node) -> &mut [u8] {
        debug_assert_eq!(self.block, node.last_metadata_block());
        let start = (node.inline_offset() % BLOCK_SIZE as u64) as usize;
        &mut self.data[start..start + node.inline_size]
    }

    pub fn write_inode(&mut self, node: &Node, node_index: usize) {
        self.write_at(
            node.inode_offset(),
            &inode(node, node_index)[..node.inode_size()],
        );
        self.write_at(node.inode_offset() + node.inode_size() as u64, &node.xattrs);
        if node.compression != Compression::None {
            self.write_at(node.map_offset(), &node.map_header());
        }
    }

    pub fn write_at(&mut self, offset: u64, data: &[u8]) {
        let base = u64::from(self.block) * BLOCK_SIZE as u64;
        let skip = base.saturating_sub(offset).min(data.len() as u64) as usize;
        let start = offset.saturating_sub(base).min(BLOCK_SIZE as u64) as usize;
        let len = (BLOCK_SIZE - start).min(data.len() - skip);
        debug_assert!(
            len == 0 || start >= self.start,
            "cannot rewrite a flushed prefix"
        );
        self.data[start..start + len].copy_from_slice(&data[skip..skip + len]);
    }
}

pub(super) struct Entry {
    pub name: Vec<u8>,
    pub node_index: usize,
    pub kind: DirentFileType,
}

pub(super) enum Content {
    File,
    Packed,
    Directory { parent: usize, entries: Vec<Entry> },
    Symlink,
    Special(SpecialFile),
}

impl Content {
    pub fn file_type(&self) -> DirentFileType {
        match self {
            Self::File | Self::Packed => DirentFileType::RegularFile,
            Self::Directory { .. } => DirentFileType::Directory,
            Self::Symlink => DirentFileType::Symlink,
            Self::Special(SpecialFile::Fifo) => DirentFileType::Fifo,
            Self::Special(SpecialFile::Socket) => DirentFileType::Socket,
            Self::Special(SpecialFile::CharacterDevice { .. }) => DirentFileType::CharacterDevice,
            Self::Special(SpecialFile::BlockDevice { .. }) => DirentFileType::BlockDevice,
        }
    }
}

// NIDs address 32-byte slots, including variable-sized inline records. No source
// handles or per-file tail buffers are retained; only one metadata block is pending.
// Multi-page records reserve contiguous blocks; their final suffix is reusable.
pub(super) struct Node {
    pub metadata: Metadata,
    pub content: Content,
    pub size: u64,
    pub nlink: u32,
    // On-disk i_u: a flat data address, or the compressed physical block count.
    pub data_block: u32,
    pub nid: u64,
    pub inline_size: usize,
    pub compact: bool,
    pub compression: Compression,
    pub big_pcluster: bool,
    pub fragment_offset: Option<u64>,
    // At most one new packed data block is allocated by a <=4 KiB fragment.
    pub fragment_block: Option<u32>,
    pub xattr_size: usize,
    // Only directory attributes survive an append; file attributes are streamed.
    pub xattrs: Vec<u8>,
}

impl Node {
    pub fn new(metadata: &Metadata, content: Content, size: u64) -> Result<Self> {
        metadata.validate()?;
        if let Content::Special(kind) = content {
            kind.validate()?;
        }
        let nlink = if matches!(content, Content::Directory { .. }) {
            2
        } else {
            1
        };
        let xattrs = xattrs(&metadata.xattrs)?;
        Ok(Self {
            metadata: Metadata {
                mode: metadata.mode,
                uid: metadata.uid,
                gid: metadata.gid,
                modified: metadata.modified,
                xattrs: Xattrs::new(),
            },
            nlink,
            content,
            size,
            data_block: 0,
            nid: 0,
            inline_size: 0,
            compact: false,
            compression: Compression::None,
            big_pcluster: false,
            fragment_offset: None,
            fragment_block: None,
            xattr_size: xattrs.len(),
            xattrs,
        })
    }

    pub fn select_format(&mut self, options: &Options, nlink: u32) -> Result<()> {
        let fits = self.size <= u64::from(u32::MAX)
            && self.metadata.uid <= u32::from(u16::MAX)
            && self.metadata.gid <= u32::from(u16::MAX)
            && nlink <= u32::from(u16::MAX)
            // Legacy Compact inodes inherit the complete superblock timestamp.
            && self.metadata.modified == options.build_time;
        if options.inode_format == InodeFormat::Compact && !fits {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "inode metadata does not fit Compact format",
            )
            .into());
        }
        self.compact = options.inode_format != InodeFormat::Extended && fits;
        Ok(())
    }

    pub fn inode_size(&self) -> usize {
        if self.compact {
            InodeCompact::size()
        } else {
            InodeExtended::size()
        }
    }

    pub fn inode_offset(&self) -> u64 {
        self.nid * INODE_SLOT_SIZE as u64
    }

    pub fn metadata_block(&self) -> u32 {
        (self.inode_offset() / BLOCK_SIZE as u64) as u32
    }

    pub fn inline_offset(&self) -> u64 {
        self.inode_offset() + self.inode_size() as u64 + self.xattr_size as u64
    }

    pub fn map_offset(&self) -> u64 {
        self.inline_offset().next_multiple_of(8)
    }

    pub fn index_offset(&self) -> u64 {
        self.map_offset() + 8
    }

    /// Logical indexes in 16-entry, 32-byte packs. Leading/trailing indexes use
    /// two-entry, 8-byte packs. NIDs are 32-byte aligned, so placement preserves this.
    pub fn two_byte_indexes(&self) -> core::ops::Range<u64> {
        let count = self.size.div_ceil(BLOCK_SIZE as u64);
        let initial = (((32 - self.index_offset() % 32) / 4) & 7).min(count);
        initial..initial + (count - initial) / 16 * 16
    }

    pub fn index_end(&self) -> u64 {
        let middle = self.two_byte_indexes();
        let bytes = self.size.div_ceil(BLOCK_SIZE as u64) * 4 - (middle.end - middle.start) * 2;
        self.index_offset() + bytes.next_multiple_of(8)
    }

    pub fn map_header(&self) -> [u8; 8] {
        let mut header = [0; 8];
        let mut fields = &mut header[..];
        fields.put_u32_le(
            self.fragment_offset
                .map_or((self.inline_size as u32) << 16, |offset| offset as u32),
        );
        let mut advise = ADVISE_COMPACTED_2B;
        if self.big_pcluster {
            advise |= ADVISE_BIG_PCLUSTER;
        }
        if self.inline_size != 0 {
            advise |= ADVISE_INLINE_TAIL;
        }
        if self.fragment_offset.is_some() {
            advise |= ADVISE_FRAGMENT;
        }
        fields.put_u16_le(advise);
        fields.put_u8(self.compression.algorithm().unwrap());
        fields.put_u8(0);
        header
    }

    pub fn metadata_end(&self) -> u64 {
        if matches!(self.content, Content::Packed) {
            self.inline_offset() + self.size.div_ceil(BLOCK_SIZE as u64) * 4
        } else if self.compression != Compression::None {
            self.index_end() + self.inline_size as u64
        } else {
            self.inline_offset() + self.inline_size as u64
        }
    }

    pub fn last_metadata_block(&self) -> u32 {
        ((self.metadata_end() - 1) / BLOCK_SIZE as u64) as u32
    }

    pub fn flat_inline_size(&self) -> usize {
        let tail = (self.size % BLOCK_SIZE as u64) as usize;
        let available =
            (BLOCK_SIZE - (self.inode_size() + self.xattr_size) % BLOCK_SIZE) % BLOCK_SIZE;
        if tail <= available { tail } else { 0 }
    }

    pub fn record_size(&self) -> u64 {
        (self.metadata_end() - self.inode_offset()).next_multiple_of(INODE_SLOT_SIZE as u64)
    }

    pub fn metadata_gap(&self) -> Option<(u64, u32)> {
        let end = self.metadata_end().next_multiple_of(INODE_SLOT_SIZE as u64);
        let free = (BLOCK_SIZE as u64 - end % BLOCK_SIZE as u64) % BLOCK_SIZE as u64;
        (free != 0).then_some((free, self.last_metadata_block()))
    }

    /// Write headers/attributes now; indexes and inline data follow incrementally.
    pub fn header_blocks(&self) -> core::ops::RangeInclusive<u32> {
        let last = if self.compression != Compression::None {
            (self.index_offset() / BLOCK_SIZE as u64) as u32
        } else {
            self.last_metadata_block()
        };
        self.metadata_block()..=last
    }

    pub fn external_size(&self) -> u64 {
        if self.compression == Compression::None {
            self.size - self.inline_size as u64
        } else {
            self.size
        }
    }
}

fn directory_blocks(mut entries: &[Entry]) -> impl Iterator<Item = &[Entry]> {
    core::iter::from_fn(move || {
        if entries.is_empty() {
            return None;
        }
        let mut size = 0;
        let count = entries
            .iter()
            .take_while(|entry| {
                size += Dirent::size() + entry.name.len();
                size <= BLOCK_SIZE
            })
            .count();
        // Validated names are at most 255 bytes, so every block makes progress.
        let (block, rest) = entries.split_at(count);
        entries = rest;
        Some(block)
    })
}

pub(super) fn directory_size(entries: &[Entry]) -> Result<u64> {
    let mut size = 0u64;
    for block in directory_blocks(entries) {
        size = size
            .div_ceil(BLOCK_SIZE as u64)
            .checked_mul(BLOCK_SIZE as u64)
            .and_then(|size| {
                size.checked_add(
                    block
                        .iter()
                        .map(|entry| (Dirent::size() + entry.name.len()) as u64)
                        .sum(),
                )
            })
            .ok_or(Error::Overflow("directory size"))?;
    }
    Ok(size)
}

fn directory_block(entries: &[Entry], nodes: &[Node]) -> [u8; BLOCK_SIZE] {
    let mut block = ZERO_BLOCK;
    let mut name_at = entries.len() * Dirent::size();
    for (index, entry) in entries.iter().enumerate() {
        let mut fields = &mut block[index * Dirent::size()..];
        fields.put_u64_le(nodes[entry.node_index].nid);
        fields.put_u16_le(name_at as u16);
        fields.put_u8(entry.kind as u8);
        fields.put_u8(0);
        block[name_at..name_at + entry.name.len()].copy_from_slice(&entry.name);
        name_at += entry.name.len();
    }
    block
}

pub(super) fn inode(node: &Node, node_index: usize) -> [u8; InodeExtended::size()] {
    let kind = match node.content {
        Content::File | Content::Packed => FileMode::IRREGULAR,
        Content::Directory { .. } => FileMode::DIR,
        Content::Symlink => FileMode::SYMLINK,
        Content::Special(SpecialFile::Fifo) => FileMode::NAMED_PIPE,
        Content::Special(SpecialFile::Socket) => FileMode::SOCKET,
        Content::Special(SpecialFile::CharacterDevice { .. }) => FileMode::CHAR_DEVICE,
        Content::Special(SpecialFile::BlockDevice { .. }) => FileMode::BLOCK_DEVICE,
    };
    let inode_data = match node.content {
        Content::Special(
            SpecialFile::CharacterDevice { major, minor }
            | SpecialFile::BlockDevice { major, minor },
        ) => (major << 8) | (minor & 0xff) | ((minor & 0xfff00) << 12),
        _ => node.data_block,
    };
    let layout = if matches!(node.content, Content::Packed) {
        Layout::ChunkBased
    } else if node.compression != Compression::None {
        Layout::CompressedCompact
    } else if node.inline_size == 0 {
        Layout::FlatPlain
    } else {
        Layout::FlatInline
    };
    let mut inode = [0; InodeExtended::size()];
    let mut fields = &mut inode[..];
    fields.put_u16_le(u16::from(!node.compact) | (layout as u16) << 1);
    fields.put_u16_le(if node.xattr_size == 0 {
        0
    } else {
        ((node.xattr_size - size_of::<XattrHeader>()) / 4 + 1) as u16
    });
    fields.put_u16_le(node.metadata.mode | kind.bits());
    if node.compact {
        fields.put_u16_le(node.nlink as u16);
        fields.put_u32_le(node.size as u32);
        fields.put_u32_le(0); // No modification-time delta.
        fields.put_u32_le(inode_data);
        fields.put_u32_le(node_index as u32 + 1);
        fields.put_u16_le(node.metadata.uid as u16);
        fields.put_u16_le(node.metadata.gid as u16);
    } else {
        fields.put_u16_le(0);
        fields.put_u64_le(node.size);
        fields.put_u32_le(inode_data);
        fields.put_u32_le(node_index as u32 + 1);
        fields.put_u32_le(node.metadata.uid);
        fields.put_u32_le(node.metadata.gid);
        fields.put_i64_le(node.metadata.modified.0);
        fields.put_u32_le(node.metadata.modified.1);
        fields.put_u32_le(node.nlink);
    }
    inode
}

// Directory records are allocated after payloads, starting in the superblock's
// unused space. File records have already been streamed in packed metadata blocks.
pub(super) fn metadata_blocks(
    nodes: &[Node],
    attributes: &Attributes,
) -> impl Iterator<Item = MetadataBlock> {
    directory_data_blocks(nodes)
        .chain(directory_inode_blocks(nodes))
        .chain(packed_inode_blocks(nodes, attributes))
        .chain(attributes.blocks())
}

fn packed_inode_blocks(
    nodes: &[Node],
    attributes: &Attributes,
) -> impl Iterator<Item = MetadataBlock> {
    nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| matches!(n.content, Content::Packed))
        .flat_map(move |(index, node)| {
            // Addresses come from existing per-file metadata; no packed index or
            // fragment payload grows in memory with the file/image size.
            let mut blocks = nodes.iter().filter_map(|n| n.fragment_block).chain(
                attributes.table_block
                    ..attributes.table_block + attributes.table.len().div_ceil(BLOCK_SIZE) as u32,
            );
            (node.metadata_block()..=node.last_metadata_block()).map(move |block| {
                let mut page = MetadataBlock::new(block);
                page.write_inode(node, index);
                let base = u64::from(block) * BLOCK_SIZE as u64;
                for at in (node.inline_offset().max(base)
                    ..node.metadata_end().min(base + BLOCK_SIZE as u64))
                    .step_by(4)
                {
                    page.write_at(at, &blocks.next().unwrap().to_le_bytes());
                }
                page
            })
        })
}

fn directory_data_blocks(nodes: &[Node]) -> impl Iterator<Item = MetadataBlock> {
    nodes
        .iter()
        .filter_map(|node| match &node.content {
            Content::Directory { entries, .. } => Some((node, entries.as_slice())),
            _ => None,
        })
        .flat_map(move |(node, entries)| {
            let blocks = node.external_size().div_ceil(BLOCK_SIZE as u64);
            directory_blocks(entries)
                .take(blocks as usize)
                .enumerate()
                .map(move |(index, entries)| MetadataBlock {
                    block: node.data_block + index as u32,
                    start: 0,
                    data: directory_block(entries, nodes),
                })
        })
}

fn directory_inode_blocks(nodes: &[Node]) -> impl Iterator<Item = MetadataBlock> {
    // Directories can occupy old file-page suffixes, so NIDs need not increase.
    // Write in allocation order: later records in a page always follow earlier
    // ones. Only block zero must be assembled in full for the superblock CRC.
    let directories = || {
        nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| matches!(n.content, Content::Directory { .. }))
    };
    let write = move |page: &mut MetadataBlock, index, node: &Node| {
        page.write_inode(node, index);
        if page.block == node.last_metadata_block()
            && node.inline_size != 0
            && let Content::Directory { entries, .. } = &node.content
        {
            let tail = directory_block(directory_blocks(entries).last().unwrap(), nodes);
            page.inline_data_mut(node)
                .copy_from_slice(&tail[..node.inline_size]);
        }
    };
    let mut first = None;
    for (index, node) in directories().filter(|(_, n)| n.metadata_block() == 0) {
        write(
            first.get_or_insert_with(|| MetadataBlock::new(0)),
            index,
            node,
        );
    }
    first.into_iter().chain(
        directories()
            .filter(|(_, n)| n.metadata_block() != 0)
            .flat_map(move |(index, node)| {
                (node.metadata_block()..=node.last_metadata_block()).map(move |block| {
                    let base = u64::from(block) * BLOCK_SIZE as u64;
                    // A reserved root's Compact-header prefix is not shared.
                    let offset = if index == 0 {
                        base
                    } else {
                        base.max(node.inode_offset())
                    };
                    let mut page = MetadataBlock::from_offset(offset);
                    write(&mut page, index, node);
                    page
                })
            }),
    )
}

pub(super) fn padding(size: u64) -> &'static [u8] {
    let padding = (BLOCK_SIZE as u64 - size % BLOCK_SIZE as u64) % BLOCK_SIZE as u64;
    &ZERO_BLOCK[..padding as usize]
}

pub(super) fn superblock(state: &State, blocks: u32) -> Vec<u8> {
    let options = &state.options;
    let inodes = state.nodes.len();
    let root_nid = state.nodes[0].nid;
    let mut superblock = vec![0; SuperBlock::size()];
    let mut fields = &mut superblock[..];
    fields.put_u32_le(MAGIC_NUMBER);
    fields.put_u32_le(0); // Checksum is filled after the first metadata block is encoded.
    fields.put_u32_le(1); // EROFS_FEATURE_COMPAT_SB_CHKSUM.
    fields.put_u8(12);
    fields.put_u8(0);
    let wide_root = root_nid > u64::from(u16::MAX);
    fields.put_u16_le(if wide_root { 0 } else { root_nid as u16 });
    fields.put_u64_le(inodes as u64);
    // Creation time is epoch + build_time delta. Keep the delta zero; compact
    // inodes inherit this timestamp, while extended inode mtimes are independent.
    fields.put_i64_le(options.build_time.0);
    fields.put_u32_le(options.build_time.1);
    fields.put_u32_le(blocks);
    fields.put_u32_le(0); // NIDs are relative to the start of the image.
    fields.put_u32_le(0); // Shared IDs address original inline records from byte zero.
    fields.put_slice(&options.uuid);
    fields.put_bytes(0, 16); // No volume label.
    let compression = options.compression;
    let packed_nid = state
        .nodes
        .iter()
        .find(|n| matches!(n.content, Content::Packed))
        .map_or(0, |n| n.nid);
    let mut features = 0;
    if compression != Compression::None {
        features |= FEATURE_ZERO_PADDING;
    }
    if options.config_size() != 0 {
        features |= FEATURE_COMPRESSION_CONFIG;
    }
    if packed_nid != 0 {
        features |= FEATURE_CHUNKED_FILE | FEATURE_FRAGMENTS;
    }
    if !state.attributes.table.is_empty() {
        features |= FEATURE_XATTR_PREFIXES;
    }
    if state
        .nodes
        .iter()
        .any(|n| n.compression != Compression::None && n.inline_size != 0)
    {
        features |= FEATURE_INLINE_TAILS;
    }
    if wide_root {
        features |= FEATURE_48BIT;
    }
    fields.put_u32_le(features);
    fields.put_u16_le(match compression {
        Compression::None => 0,
        Compression::Lz4 if options.config_size() == 0 => u16::MAX, // Legacy maximum distance.
        _ => 1 << compression.algorithm().unwrap(),
    });
    fields.put_u32_le(0); // No additional devices or device table.
    fields.put_u8(0); // No directory block override.
    fields.put_u8(state.attributes.prefixes.len() as u8);
    fields.put_u32_le(state.attributes.prefix_start);
    fields.put_u64_le(packed_nid);
    fields.put_bytes(0, 8); // Xattr filter fields and zero build-time delta.
    fields.put_u64_le(if wide_root { root_nid } else { 0 }); // root_nid_wide.
    // Global codec configuration follows the base superblock. It is written with
    // the final header and included in its CRC, even when all root metadata moved.
    match compression {
        Compression::Lz4 if options.config_size() != 0 => {
            superblock.put_u16_le(14);
            superblock.put_u16_le(u16::MAX);
            superblock.put_u16_le((options.pcluster_size / BLOCK_SIZE as u32) as u16);
            superblock.put_bytes(0, 10);
        }
        Compression::None | Compression::Lz4 => {}
        Compression::Lzma => {
            superblock.put_u16_le(14);
            superblock.put_u32_le(LZMA_DICT_SIZE);
            superblock.put_bytes(0, 10); // MicroLZMA format 0 and reserved fields.
        }
        Compression::Deflate => {
            superblock.put_u16_le(6);
            superblock.put_u8(15); // Raw DEFLATE, 32 KiB window.
            superblock.put_bytes(0, 5);
        }
        Compression::Zstd => {
            superblock.put_u16_le(6);
            superblock.put_u8(0); // Standard Zstd frame format.
            superblock.put_u8(ZSTD_WINDOW_LOG - 10);
            superblock.put_bytes(0, 4);
        }
    }
    // A relocated root leaves the suffix after the header/configuration zeroed.
    if root_nid >= (BLOCK_SIZE / INODE_SLOT_SIZE) as u64 {
        set_superblock_checksum(&mut superblock, &ZERO_BLOCK);
    }
    superblock
}

pub(super) fn set_superblock_checksum(superblock: &mut [u8], first_block: &[u8; BLOCK_SIZE]) {
    // For 4 KiB blocks, EROFS covers image bytes 1024..4096, including packed
    // directory metadata and padding, with the checksum field treated as zero.
    // Later hard-link patches only affect non-directory inodes outside block 0.
    superblock[4..8].fill(0);
    // The crate applies a final XOR; EROFS stores the uncomplemented CRC state.
    let suffix = SUPER_BLOCK_OFFSET as usize + superblock.len();
    let crc = !crc32c_append(crc32c(superblock), &first_block[suffix..]);
    (&mut superblock[4..]).put_u32_le(crc);
}
