//! Bounded compression and Compact indexes; both I/O drivers share decisions.

use alloc::vec::Vec;
use bytes::BufMut;
#[cfg(any(feature = "lz4", feature = "lzma", feature = "deflate"))]
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

pub(super) struct Compressor {
    compression: Compression,
    scratch: Vec<u8>,
    pub data: Vec<u8>,
    pub placement: Placement,
    limit: usize,
    inline_capacity: usize,
    big_pcluster: bool,
    fragments: bool,
    // A completed pending pack plus this extent's Compact entries.
    pub indexes: Vec<u8>,
    pack: [u8; 32],
    pack_len: usize,
    next_index: u64,
    index_count: u64,
    two_byte_indexes: core::ops::Range<u64>,
}

impl Compressor {
    pub fn new(node: &Node, options: &Options) -> Self {
        Self {
            compression: node.compression,
            scratch: Vec::new(),
            data: Vec::new(),
            placement: Placement::External,
            limit: options.pcluster_size as usize,
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
            two_byte_indexes: node.two_byte_indexes(),
        }
    }

    /// Consumes a block-aligned prefix or the final tail. Data and Compact indexes
    /// are emitted incrementally; an unfinished index pack survives across calls.
    pub fn encode(&mut self, input: &[u8], block: u32, fragment_offset: u64) -> Result<usize> {
        debug_assert!(!input.is_empty() && input.len() <= INPUT_SIZE);
        let final_input =
            self.next_index + input.len().div_ceil(BLOCK_SIZE) as u64 == self.index_count;
        self.indexes.clear();
        let (consumed, compressed) = self.encode_data(input, final_input, fragment_offset)?;
        self.write_indexes(consumed, compressed, block);
        Ok(consumed)
    }

    fn encode_data(
        &mut self,
        input: &[u8],
        final_input: bool,
        fragment_offset: u64,
    ) -> Result<(usize, bool)> {
        self.data.clear();
        self.placement = Placement::External;
        let try_inline = final_input && self.inline_capacity != 0;
        let mut consumed = if try_inline {
            input.len()
        } else {
            input.len() / BLOCK_SIZE * BLOCK_SIZE
        };

        loop {
            let inline_candidate = try_inline && consumed == input.len();
            let encoded = if consumed > BLOCK_SIZE || inline_candidate {
                self.compress(&input[..consumed])?
                    // EROFS skips leading zeros. A zero-leading DEFLATE stored
                    // block cannot be distinguished from padding; leave it PLAIN.
                    .filter(|_| self.scratch[0] != 0)
            } else {
                None
            };

            if inline_candidate {
                if let Some(len) = encoded
                    && len < input.len()
                    && len <= self.inline_capacity
                {
                    self.data.extend_from_slice(&self.scratch[..len]);
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

            // Reuse the same encoded candidate when inline storage did not fit.
            // External extents must end on a logical-block boundary.
            if consumed.is_multiple_of(BLOCK_SIZE)
                && let Some(len) = encoded
                && len <= self.limit
                && len.div_ceil(BLOCK_SIZE) < consumed / BLOCK_SIZE
            {
                self.data.resize(len.next_multiple_of(BLOCK_SIZE) - len, 0);
                self.data.extend_from_slice(&self.scratch[..len]);
                return Ok((consumed, true));
            }
            if consumed <= BLOCK_SIZE {
                break;
            }
            // A rejected short tail first retries its aligned prefix. Otherwise
            // halve the candidate; every iteration encodes strictly less input.
            // ponytail: prefix halving may miss a better fit; use destination-size
            // compression if measured ratios justify a different encoder.
            consumed = if consumed.is_multiple_of(BLOCK_SIZE) {
                consumed / 2 / BLOCK_SIZE * BLOCK_SIZE
            } else {
                consumed / BLOCK_SIZE * BLOCK_SIZE
            };
        }

        let consumed = input.len().min(BLOCK_SIZE);
        self.data.extend_from_slice(&input[..consumed]);
        self.data.resize(BLOCK_SIZE, 0);
        Ok((consumed, false))
    }

    fn write_indexes(&mut self, consumed: usize, compressed: bool, block: u32) {
        let count = consumed.div_ceil(BLOCK_SIZE);
        let blocks = if self.placement == Placement::Fragment {
            0
        } else {
            self.data.len().div_ceil(BLOCK_SIZE) as u32
        };
        for index in 0..count {
            let (slots, bits, bytes) = if self.two_byte_indexes.contains(&self.next_index) {
                (16, 14, 32)
            } else {
                (2, 16, 8)
            };
            if self.pack_len == 0 {
                self.pack.fill(0);
                // Legacy predictors name the physical block preceding the pack's
                // first HEAD, or the current extent if the pack starts in NONHEADs.
                let base = if self.big_pcluster {
                    block + if index > 1 { blocks } else { 0 }
                } else {
                    block.wrapping_sub(u32::from(index == 0))
                };
                (&mut self.pack[bytes - 4..bytes]).put_u32_le(base);
            }
            let value = if index == 0 {
                u32::from(compressed) << 12 // HEAD1 or PLAIN, cluster offset zero.
            } else if self.big_pcluster && index == 1 {
                2 << 12 | 1 << 11 | blocks
            } else {
                // Only the last NONHEAD in a pack stores the forward distance.
                2 << 12
                    | if self.pack_len + 1 == slots {
                        count - index
                    } else {
                        index
                    } as u32
            };
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

    // None means the encoder filled its bounded output before finishing.
    fn compress(&mut self, input: &[u8]) -> Result<Option<usize>> {
        let len = match self.compression {
            #[cfg(feature = "lz4")]
            Compression::Lz4 => {
                self.scratch
                    .resize(lz4_flex::block::get_maximum_output_size(input.len()), 0);
                lz4_flex::block::compress_into(input, &mut self.scratch)
                    .map_err(io::Error::other)?
            }
            #[cfg(feature = "lzma")]
            Compression::Lzma => {
                use lzma_rust2::{LzmaOptions, LzmaWriter};
                use std::io::Write;

                self.scratch.resize(self.limit, 0);
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
                let mut encoder = CompressorOxide::new(create_comp_flags_from_zip_params(6, 0, 0));
                let (status, read, written) =
                    compress(&mut encoder, input, &mut self.scratch, TDEFLFlush::Finish);
                match status {
                    TDEFLStatus::Done if read == input.len() => written,
                    TDEFLStatus::Okay => return Ok(None),
                    _ => return Err(io::Error::other("DEFLATE encoding failed").into()),
                }
            }
            #[cfg(feature = "zstd")]
            Compression::Zstd => {
                // ruzstd's encoder unwraps I/O errors: only give it infallible
                // slice/Vec I/O, never the caller's source or image output.
                self.scratch = ruzstd::encoding::compress_to_vec(
                    input,
                    ruzstd::encoding::CompressionLevel::Fastest,
                );
                self.scratch.len()
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
