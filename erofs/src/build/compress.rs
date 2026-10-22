//! Bounded compression and Compact indexes; both I/O drivers share decisions.

#[cfg(any(feature = "lz4", feature = "deflate", feature = "zstd"))]
use alloc::boxed::Box;
use alloc::vec::Vec;
use bytes::BufMut;
use std::io;

use super::{
    Compression, Options,
    encode::{BLOCK_SIZE, Node},
};
use crate::{Error, Result};

pub(super) const INPUT_SIZE: usize = 64 * 1024;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Placement {
    External,
    Inline,
    Fragment,
}

pub(super) struct BufferedFile {
    pub data: Vec<u8>,
    pub indexes: Vec<u8>,
}

pub(super) struct EncodedExtent<'a> {
    pub consumed: usize,
    pub data: &'a [u8],
    pub indexes: &'a [u8],
    pub placement: Placement,
}

pub(super) struct Compressor {
    data: Vec<u8>,
    placement: Placement,
    inline_capacity: usize,
    big_pcluster: bool,
    fragments: bool,
    // A completed pending pack plus this extent's Compact entries.
    indexes: Vec<u8>,
    pack: [u8; 32],
    pack_len: usize,
    next_index: u64,
    index_count: u64,
    position: u64,
    file_size: u64,
    hint: usize,
    two_byte_indexes: core::ops::Range<u64>,
}

impl Compressor {
    pub fn new(node: &Node, options: &Options) -> Self {
        Self {
            data: Vec::new(),
            placement: Placement::External,
            inline_capacity: if options.inline_compressed_tails {
                ((BLOCK_SIZE as u64 - node.index_end() % BLOCK_SIZE as u64) % BLOCK_SIZE as u64)
                    as usize
            } else {
                0
            },
            big_pcluster: node.big_pcluster,
            fragments: options.fragments,
            indexes: Vec::with_capacity(32 + INPUT_SIZE / BLOCK_SIZE * 4),
            pack: [0; 32],
            pack_len: 0,
            next_index: 0,
            index_count: node.size.div_ceil(BLOCK_SIZE as u64),
            position: 0,
            file_size: node.size,
            hint: 4 * BLOCK_SIZE,
            two_byte_indexes: node.two_byte_indexes(),
        }
    }

    // A complete short file can fall back to flat storage without rereading
    // input or output. Larger files retain the streaming extent path.
    pub fn encode_file(
        mut self,
        encoder: &mut Encoder,
        mut input: &[u8],
        mut block: u32,
    ) -> Result<BufferedFile> {
        debug_assert_eq!(self.position, 0);
        debug_assert_eq!(input.len() as u64, self.file_size);
        debug_assert!(input.len() <= INPUT_SIZE && self.inline_capacity == 0 && !self.fragments);
        let mut file = BufferedFile {
            data: Vec::with_capacity(input.len().next_multiple_of(BLOCK_SIZE)),
            indexes: Vec::new(),
        };
        while !input.is_empty() {
            let extent = self.encode(encoder, input, block, 0)?;
            file.data.extend_from_slice(extent.data);
            file.indexes.extend_from_slice(extent.indexes);
            block += (extent.data.len() / BLOCK_SIZE) as u32;
            input = &input[extent.consumed..];
        }
        Ok(file)
    }

    /// Consumes a byte-granular prefix. Complete Compact packs stream out;
    /// a partial pack and a logical cluster shared by two extents survive calls.
    pub fn encode(
        &mut self,
        encoder: &mut Encoder,
        input: &[u8],
        block: u32,
        fragment_offset: u64,
    ) -> Result<EncodedExtent<'_>> {
        debug_assert!(!input.is_empty() && input.len() <= INPUT_SIZE);
        let final_input = self.position + input.len() as u64 == self.file_size;
        self.indexes.clear();
        let (consumed, compressed) =
            self.encode_data(encoder, input, final_input, fragment_offset)?;
        self.write_indexes(consumed, compressed, block);
        Ok(EncodedExtent {
            consumed,
            data: &self.data,
            indexes: &self.indexes,
            placement: self.placement,
        })
    }

    pub fn inline_tail(&self) -> Option<&[u8]> {
        (self.placement == Placement::Inline).then_some(self.data.as_slice())
    }

    fn encode_data(
        &mut self,
        encoder: &mut Encoder,
        input: &[u8],
        final_input: bool,
        fragment_offset: u64,
    ) -> Result<(usize, bool)> {
        self.data.clear();
        self.placement = Placement::External;
        let try_inline = final_input && self.inline_capacity != 0;
        let mut consumed = if try_inline || encoder.limit >= INPUT_SIZE / 4 {
            input.len()
        } else {
            input.len().min(self.hint)
        };
        let mut low = BLOCK_SIZE;
        let mut high = input.len() + 1;
        let mut best = 0;

        // Favor image size: stop within 8 bytes of the destination capacity.
        // shortcut: cap probes at 20; use destination-size codecs if encoding dominates.
        // Prefix sizes need not be monotonic.
        for _ in 0..20 {
            let encoded = if consumed > BLOCK_SIZE || try_inline {
                encoder
                    .compress(&input[..consumed])?
                    // EROFS skips leading zeros; a zero-leading DEFLATE stored
                    // block is indistinguishable from padding. Keep it PLAIN.
                    .filter(|_| encoder.scratch[0] != 0)
            } else {
                None
            };

            if try_inline && consumed == input.len() {
                if let Some(len) = encoded
                    && len < input.len()
                    && len <= self.inline_capacity
                {
                    self.data.extend_from_slice(&encoder.scratch[..len]);
                    self.placement = Placement::Inline;
                    return Ok((input.len(), true));
                }
                if input.len() <= self.inline_capacity {
                    self.data.extend_from_slice(input);
                    self.placement = Placement::Inline;
                    return Ok((input.len(), false));
                }
            }

            if final_input
                && input.len() <= BLOCK_SIZE
                && self.fragments
                && fragment_offset + input.len() as u64 <= u64::from(u32::MAX)
            {
                self.data.extend_from_slice(input);
                self.placement = Placement::Fragment;
                return Ok((input.len(), true));
            }
            if consumed <= BLOCK_SIZE {
                break;
            }

            let estimate = if let Some(len) = encoded {
                let estimate = ((consumed as u64 * encoder.limit as u64 / len as u64)
                    .min(INPUT_SIZE as u64)) as usize;
                if len <= encoder.limit {
                    if len.next_multiple_of(BLOCK_SIZE) < consumed {
                        self.data.clear();
                        self.data.extend_from_slice(&encoder.scratch[..len]);
                        best = consumed;
                        self.hint = estimate.max(BLOCK_SIZE + 1);
                    }
                    low = consumed;
                    if consumed == input.len() || encoder.limit - len <= 8 {
                        break;
                    }
                } else {
                    high = consumed;
                }
                estimate
            } else {
                high = consumed;
                (low + high) / 2
            };
            if high <= low + 1 {
                break;
            }
            // Heterogeneous prefixes can predict outside the bracket. Bisect
            // instead of alternating one-byte steps at its opposite edges.
            consumed = if best == 0 && estimate <= low {
                // Quickly reject incompressible input without twenty probes.
                low + 1
            } else if estimate > low && estimate < high {
                estimate
            } else {
                (low + high) / 2
            };
        }

        if best != 0 {
            let len = self.data.len();
            let padded = len.next_multiple_of(BLOCK_SIZE);
            self.data.resize(padded, 0);
            self.data.copy_within(..len, padded - len);
            self.data[..padded - len].fill(0);
            return Ok((best, true));
        }

        let consumed = input.len().min(BLOCK_SIZE);
        self.data.extend_from_slice(&input[..consumed]);
        self.data.resize(BLOCK_SIZE, 0);
        Ok((consumed, false))
    }

    fn write_indexes(&mut self, consumed: usize, compressed: bool, block: u32) {
        let offset = (self.position % BLOCK_SIZE as u64) as usize;
        let whole = (offset + consumed) / BLOCK_SIZE;
        let count = whole.max(1);
        let blocks = if self.placement == Placement::Fragment {
            0
        } else {
            self.data.len().div_ceil(BLOCK_SIZE) as u32
        };
        for index in 0..count {
            // A pack starting in NONHEADs predicts the current extent, or the
            // following physical block after a big-pcluster count entry.
            let base = if self.big_pcluster {
                block + if index > 1 { blocks } else { 0 }
            } else {
                block.wrapping_sub(u32::from(index == 0))
            };
            let slots = if self.two_byte_indexes.contains(&self.next_index) {
                16
            } else {
                2
            };
            let value = if index == 0 {
                u32::from(compressed) << 12 | offset as u32
            } else if self.big_pcluster && index == 1 {
                2 << 12 | 1 << 11 | blocks
            } else {
                // The final NONHEAD in a pack stores a forward distance.
                2 << 12
                    | if self.pack_len + 1 == slots {
                        count - index
                    } else {
                        index
                    } as u32
            };
            self.write_index(value, base);
        }
        self.position += consumed as u64;
        // A final partial logical cluster belongs to the preceding extent.
        // A PLAIN sentinel at EOF terminates it; it never references file data.
        if whole != 0
            && self.position == self.file_size
            && !self.position.is_multiple_of(BLOCK_SIZE as u64)
        {
            self.write_index(
                (self.position % BLOCK_SIZE as u64) as u32,
                (block + blocks).wrapping_sub(u32::from(!self.big_pcluster)),
            );
        }
    }

    fn write_index(&mut self, value: u32, base: u32) {
        debug_assert!(self.next_index < self.index_count);
        let (slots, bits, bytes) = if self.two_byte_indexes.contains(&self.next_index) {
            (16, 14, 32)
        } else {
            (2, 16, 8)
        };
        if self.pack_len == 0 {
            self.pack.fill(0);
            (&mut self.pack[bytes - 4..bytes]).put_u32_le(base);
        }
        let bit = self.pack_len * bits;
        for (i, byte) in (value << (bit % 8)).to_le_bytes()[..3].iter().enumerate() {
            self.pack[bit / 8 + i] |= byte;
        }
        self.pack_len += 1;
        self.next_index += 1;
        if self.pack_len == slots || self.next_index == self.index_count {
            self.indexes.extend_from_slice(&self.pack[..bytes]);
            self.pack_len = 0;
        }
    }
}

// Builder-owned allocations; file positions and indexes stay in Compressor.
pub(super) struct Encoder {
    compression: Compression,
    limit: usize,
    scratch: Vec<u8>,
    #[cfg(feature = "lz4")]
    lz4_small: Option<Box<lz4_flex::block::CompressTable>>,
    #[cfg(feature = "lz4")]
    lz4_large: Option<Box<lz4_flex::block::CompressTable>>,
    #[cfg(feature = "deflate")]
    deflate: Option<Box<miniz_oxide::deflate::core::CompressorOxide>>,
    #[cfg(feature = "zstd")]
    zstd: Option<Box<zrip::CompressContext>>,
}

impl Encoder {
    pub fn new(options: &Options) -> Self {
        Self {
            compression: options.compression,
            limit: options.pcluster_size as usize,
            scratch: Vec::new(),
            #[cfg(feature = "lz4")]
            lz4_small: None,
            #[cfg(feature = "lz4")]
            lz4_large: None,
            #[cfg(feature = "deflate")]
            deflate: None,
            #[cfg(feature = "zstd")]
            zstd: None,
        }
    }

    // None means the encoder filled its bounded output before finishing.
    fn compress(&mut self, input: &[u8]) -> Result<Option<usize>> {
        debug_assert!(input.len() <= INPUT_SIZE);
        let len = match self.compression {
            #[cfg(feature = "lz4")]
            Compression::Lz4 => {
                use lz4_flex::block::{
                    CompressTable, compress_into_with_table, get_maximum_output_size,
                };

                // Keep both widths so a previous large input cannot change small-input output.
                let small = input.len() < usize::from(u16::MAX);
                let table = if small {
                    &mut self.lz4_small
                } else {
                    &mut self.lz4_large
                };
                let table = table.get_or_insert_with(|| {
                    Box::new(if small {
                        CompressTable::small()
                    } else {
                        CompressTable::large()
                    })
                });
                self.scratch.resize(get_maximum_output_size(input.len()), 0);
                // compress_into_with_table clears history on every call.
                compress_into_with_table(input, &mut self.scratch, table)
                    .map_err(io::Error::other)?
            }
            #[cfg(feature = "lzma")]
            Compression::Lzma => {
                use lzma_rust2::{LzmaOptions, LzmaWriter};
                use std::io::Write;

                // Keep a bounded oversized result so the fit search can use its
                // measured ratio instead of repeatedly bisecting on WriteZero.
                self.scratch.resize(self.limit.max(input.len() * 2), 0);
                let mut output = io::Cursor::new(self.scratch.as_mut_slice());
                let mut options = LzmaOptions::with_preset(1);
                options.dict_size = super::encode::LZMA_DICT_SIZE;
                let mut encoder = LzmaWriter::new(
                    &mut output,
                    &options,
                    false,
                    false,
                    Some(input.len() as u64),
                )?;
                let props = encoder.props();
                match encoder
                    .write_all(input)
                    .and_then(|()| encoder.finish().map(|_| ()))
                {
                    Ok(()) => {}
                    Err(error) if error.kind() == io::ErrorKind::WriteZero => return Ok(None),
                    Err(error) => return Err(error.into()),
                }
                let len = output.position() as usize;
                // MicroLZMA is raw LZMA1 without an end marker, replacing its
                // initial zero byte with the complemented properties byte.
                debug_assert!(len >= 5 && self.scratch[0] == 0);
                self.scratch[0] = !props;
                len
            }
            #[cfg(feature = "deflate")]
            Compression::Deflate => {
                use miniz_oxide::deflate::core::{
                    CompressorOxide, TDEFLFlush, TDEFLStatus, compress,
                    create_comp_flags_from_zip_params,
                };

                self.scratch.resize(self.limit, 0);
                let encoder = self.deflate.get_or_insert_with(|| {
                    Box::new(CompressorOxide::new(create_comp_flags_from_zip_params(
                        6, 0, 0,
                    )))
                });
                // Reset even after a trial exhausted its output before finishing.
                encoder.reset();
                let (status, read, written) =
                    compress(encoder, input, &mut self.scratch, TDEFLFlush::Finish);
                match status {
                    TDEFLStatus::Done if read == input.len() => written,
                    TDEFLStatus::Okay => return Ok(None),
                    _ => return Err(io::Error::other("DEFLATE encoding failed").into()),
                }
            }
            #[cfg(feature = "zstd")]
            Compression::Zstd => {
                // Single-segment frames advertise the input length as their window.
                const { assert!(INPUT_SIZE <= 1 << super::encode::ZSTD_WINDOW_LOG) };
                let encoder = match &mut self.zstd {
                    Some(encoder) => encoder,
                    slot @ None => slot.insert(Box::new(
                        zrip::CompressContext::new(1).map_err(io::Error::other)?,
                    )),
                };
                self.scratch.resize(zrip::compress_bound(input.len()), 0);
                // Each call starts an independent frame, retaining only allocations.
                encoder
                    .compress_into(input, &mut self.scratch)
                    .map_err(io::Error::other)?
            }
            _ => {
                return Err(Error::NotSupported(format!(
                    "compression encoder {:?} is disabled",
                    self.compression
                )));
            }
        };
        Ok(Some(len))
    }
}

#[cfg(all(test, any(feature = "lz4", feature = "deflate", feature = "zstd")))]
mod test {
    use super::*;

    #[test]
    fn reused_encoders_reset_between_trials() {
        let mut seed = 7u32;
        let random: Vec<_> = (0..INPUT_SIZE)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed as u8
            })
            .collect();
        let zeros = vec![0; INPUT_SIZE];
        let repeated: Vec<_> = random[..1024]
            .iter()
            .copied()
            .cycle()
            .take(INPUT_SIZE)
            .collect();
        for compression in [Compression::Lz4, Compression::Deflate, Compression::Zstd]
            .into_iter()
            .filter(|compression| compression.enabled())
        {
            let options = Options::default().compression(compression);
            let mut reused = Encoder::new(&options);
            let mut exhausted = false;
            for input in [
                random.as_slice(),
                &repeated[..8192],
                &repeated[..65535],
                &random[..32768],
                &zeros[..4097],
                &repeated[..65534],
                &zeros[..65536],
                &repeated[..16384],
                &random[..4097],
                &repeated[..16384],
            ] {
                let mut fresh = Encoder::new(&options);
                let actual = reused.compress(input).unwrap();
                assert_eq!(actual, fresh.compress(input).unwrap());
                let Some(len) = actual else {
                    exhausted = true;
                    continue;
                };
                let encoded = &reused.scratch[..len];
                assert_eq!(encoded, &fresh.scratch[..len]);
                match compression {
                    #[cfg(feature = "lz4")]
                    Compression::Lz4 => {
                        // Preserve the one-shot API's choice of table width as well as bytes.
                        let mut expected =
                            vec![0; lz4_flex::block::get_maximum_output_size(input.len())];
                        let len = lz4_flex::block::compress_into(input, &mut expected).unwrap();
                        assert_eq!(encoded, &expected[..len]);
                        let mut decoded = vec![0; input.len()];
                        assert_eq!(
                            lz4_flex::block::decompress_into(encoded, &mut decoded).unwrap(),
                            input.len()
                        );
                        assert_eq!(decoded, input);
                    }
                    #[cfg(feature = "deflate")]
                    Compression::Deflate => {
                        assert_eq!(encoded, miniz_oxide::deflate::compress_to_vec(input, 6));
                        assert_eq!(
                            miniz_oxide::inflate::decompress_to_vec(encoded).unwrap(),
                            input
                        );
                    }
                    #[cfg(feature = "zstd")]
                    Compression::Zstd => {
                        assert_eq!(encoded, zrip::compress(input, 1).unwrap());
                        assert_eq!(
                            zrip::decompress_with_limit(encoded, input.len()).unwrap(),
                            input
                        );
                    }
                    _ => unreachable!(),
                }
            }
            assert_eq!(exhausted, compression == Compression::Deflate);
        }
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn zstd_frames_reuse_context_without_carrying_history() {
        fn assert_send<T: Send>() {}
        assert_send::<zrip::CompressContext>();
        assert_send::<super::super::State>();

        let data: Vec<_> = (0..INPUT_SIZE)
            .map(|i| ((i * 31) ^ (i / 97)) as u8)
            .collect();
        let options = Options::default().compression(Compression::Zstd);
        let mut compressor = Encoder::new(&options);
        let mut other = Encoder::new(&options);
        assert!(compressor.zstd.is_none());
        for size in [16384, INPUT_SIZE, 4097, 0, 16384] {
            // Interleave files on the same thread, including different input bytes.
            other.compress(&vec![42; INPUT_SIZE - size]).unwrap();
            let input = &data[..size];
            let len = compressor.compress(input).unwrap().unwrap();
            let frame = &compressor.scratch[..len];
            assert!(compressor.zstd.is_some());
            assert_eq!(frame, zrip::compress(input, 1).unwrap());
            let header = zrip::frame::header::parse_frame_header(frame).unwrap();
            assert!(header.content_checksum);
            assert_eq!(header.frame_content_size, Some(size as u64));
            assert!(header.window_size <= 1 << super::super::encode::ZSTD_WINDOW_LOG);
            assert_eq!(zrip::decompress_with_limit(frame, size).unwrap(), input);
            let mut corrupt = frame.to_vec();
            *corrupt.last_mut().unwrap() ^= 1;
            assert!(zrip::decompress_with_limit(&corrupt, size).is_err());
        }
    }
}
