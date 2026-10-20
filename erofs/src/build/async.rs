use std::{
    borrow::Borrow,
    io::{self, SeekFrom},
};
use typed_path::UnixPath;

use super::{
    Metadata, Options, SpecialFile, State, check_output_length,
    encode::{self, BLOCK_SIZE, Content, Node, ZERO_BLOCK},
    symlink_target,
};
use crate::{
    Result,
    backend::{AsyncRead, AsyncSeek, AsyncWrite},
    types::SUPER_BLOCK_OFFSET,
};

/// Runtime-independent async counterpart of [`Builder`](super::Builder), available with `std`.
///
/// Uses the library's [`AsyncRead`], [`AsyncWrite`] and [`AsyncSeek`] backend traits,
/// following the same async-function model as `backend::AsyncImage`. Memory cursors,
/// byte-slice inputs and EROFS async file inputs work without a runtime dependency.
/// The optional `tokio` feature provides `backend::TokioIo` for Tokio types.
///
/// Path, metadata, hard-link and format rules are shared with `Builder`.
/// `append_dir` and `append_hard_link` only change metadata and remain synchronous;
/// operations performing I/O are async. The output must be empty and honor seeks,
/// not use append mode. Full data blocks are streamed; inline tails share one
/// pending metadata block rather than accumulating per-file buffers.
///
/// # Errors and cancellation
///
/// Append validation errors leave the builder usable. An I/O error or cancellation
/// of an append after I/O begins poisons it: discard the output and start again.
/// Dropping an unpolled append future has no effect. Canceling `new` or `finish`
/// may leave an incomplete output, which must also be discarded. Dropping never
/// finishes the image, and `finish` flushes but does not shut down the returned
/// writer. Futures are `Send` when their writer, reader, metadata and path arguments are.
///
/// ```
/// use std::io::Cursor;
/// use erofs_rs::build::{AsyncBuilder, Metadata};
/// # async fn example() -> erofs_rs::Result<()> {
/// let mut image = AsyncBuilder::new(Cursor::new(Vec::new())).await?;
/// image.append_dir("etc", Metadata { mode: 0o755, ..Metadata::default() })?;
/// image.append_file("etc/message", Metadata::default(), 5, &b"hello"[..]).await?;
/// image.append_symlink("message", Metadata { mode: 0o777, ..Metadata::default() }, "etc/message").await?;
/// image.append_hard_link("copy", "etc/message")?;
/// let bytes = image.finish().await?.into_inner();
/// # assert!(!bytes.is_empty());
/// # Ok(())
/// # }
/// ```
#[must_use = "await finish() to complete the image"]
pub struct AsyncBuilder<W: AsyncWrite + AsyncSeek> {
    output: W,
    state: State,
}

impl Options {
    /// Creates an async builder with these settings and an empty output.
    ///
    /// Invalid options are rejected before output I/O; dropping an unpolled future
    /// has no effect. On I/O error or cancellation discard the incomplete output.
    pub async fn build_async<W: AsyncWrite + AsyncSeek>(
        self,
        mut output: W,
    ) -> Result<AsyncBuilder<W>> {
        let state = State::new(self)?;
        if output.seek(SeekFrom::End(0)).await? != 0 {
            return Err(
                io::Error::new(io::ErrorKind::InvalidInput, "image output must be empty").into(),
            );
        }
        output.seek(SeekFrom::Start(0)).await?;
        output.write_all(&ZERO_BLOCK).await?;
        Ok(AsyncBuilder { output, state })
    }
}

impl<W: AsyncWrite + AsyncSeek> AsyncBuilder<W> {
    /// Starts an image at offset zero with default [`Options`].
    /// Nonempty outputs are rejected, not truncated.
    pub async fn new(output: W) -> Result<Self> {
        Options::default().build_async(output).await
    }

    /// Same as [`Builder::append_dir`](super::Builder::append_dir); no I/O or await is needed.
    pub fn append_dir(
        &mut self,
        path: impl AsRef<UnixPath>,
        metadata: impl Borrow<Metadata>,
    ) -> Result<()> {
        self.state.append_dir(path.as_ref(), metadata.borrow())
    }

    /// Copies exactly `size` bytes, retaining any suffix in a borrowed reader.
    /// A zero size does not read the source. Short input, read/write errors or
    /// cancellation after I/O begins poison the builder.
    pub async fn append_file(
        &mut self,
        path: impl AsRef<UnixPath>,
        metadata: impl Borrow<Metadata>,
        size: u64,
        data: impl AsyncRead,
    ) -> Result<()> {
        self.append_payload(path.as_ref(), metadata.borrow(), Content::File, size, data)
            .await
    }

    /// Same target-byte and validation rules as [`Builder::append_symlink`](super::Builder::append_symlink).
    pub async fn append_symlink(
        &mut self,
        path: impl AsRef<UnixPath>,
        metadata: impl Borrow<Metadata>,
        target: impl AsRef<UnixPath>,
    ) -> Result<()> {
        let target = symlink_target(target.as_ref())?;
        self.append_payload(
            path.as_ref(),
            metadata.borrow(),
            Content::Symlink,
            target.len() as u64,
            target,
        )
        .await
    }

    /// Same as [`Builder::append_special`](super::Builder::append_special).
    /// This may flush a metadata block, so it is async despite having no payload.
    pub async fn append_special(
        &mut self,
        path: impl AsRef<UnixPath>,
        metadata: impl Borrow<Metadata>,
        kind: SpecialFile,
    ) -> Result<()> {
        self.append_payload(
            path.as_ref(),
            metadata.borrow(),
            Content::Special(kind),
            0,
            &[][..],
        )
        .await
    }

    /// Same as [`Builder::append_hard_link`](super::Builder::append_hard_link); no I/O or await is needed.
    pub fn append_hard_link(
        &mut self,
        path: impl AsRef<UnixPath>,
        target: impl AsRef<UnixPath>,
    ) -> Result<()> {
        self.state.append_hard_link(path.as_ref(), target.as_ref())
    }

    /// Writes final metadata and the superblock, flushes, and returns the writer.
    /// On error or cancellation discard the output. Durability is backend-specific;
    /// for a Tokio file, unwrap `TokioIo` and await the file's `sync_all()`.
    pub async fn finish(mut self) -> Result<W> {
        let blocks = self.state.layout()?;
        self.flush_metadata().await?;
        self.flush_fragments().await?;
        let mut superblock = encode::superblock(&self.state, blocks);
        for (offset, block) in encode::metadata_blocks(&self.state.nodes, &self.state.attributes) {
            if offset == 0 {
                encode::set_superblock_checksum(&mut superblock, &block);
            }
            write_at(&mut self.output, offset, &block).await?;
        }
        for (node_index, node) in self.state.nodes.iter().enumerate() {
            if node.nlink > 1 && !matches!(node.content, Content::Directory { .. }) {
                write_at(
                    &mut self.output,
                    node.inode_offset(),
                    &encode::inode(node, node_index)[..node.inode_size()],
                )
                .await?;
            }
        }
        write_at(&mut self.output, SUPER_BLOCK_OFFSET, &superblock).await?;
        check_output_length(
            self.output.seek(SeekFrom::End(0)).await?,
            u64::from(blocks) * BLOCK_SIZE as u64,
        )?;
        self.output.flush().await?;
        Ok(self.output)
    }

    async fn append_payload(
        &mut self,
        path: &UnixPath,
        metadata: &Metadata,
        content: Content,
        size: u64,
        mut data: impl AsyncRead,
    ) -> Result<()> {
        let entry = self
            .state
            .prepare_payload(path, metadata, content, size, 1)?;
        self.prepare_metadata(&entry.node).await?;
        #[cfg(any(
            feature = "lz4",
            feature = "lzma",
            feature = "deflate",
            feature = "zstd"
        ))]
        if entry.node.compression != super::Compression::None {
            return self.append_compressed(entry, &mut data).await;
        }
        let external_size = entry.node.external_size();
        if external_size != 0 {
            self.output
                .seek(SeekFrom::Start(
                    u64::from(entry.node.data_block) * BLOCK_SIZE as u64,
                ))
                .await?;
            copy_exact(&mut data, &mut self.output, external_size).await?;
            self.output
                .write_all(encode::padding(external_size))
                .await?;
        }
        let page = self.state.pending_metadata.as_mut().unwrap();
        read_exact(&mut data, page.inline_data_mut(&entry.node)).await?;
        self.state.commit_payload(entry);
        Ok(())
    }

    #[cfg(any(
        feature = "lz4",
        feature = "lzma",
        feature = "deflate",
        feature = "zstd"
    ))]
    async fn append_compressed(
        &mut self,
        mut entry: super::PendingEntry,
        data: &mut impl AsyncRead,
    ) -> Result<()> {
        use super::compress::{Compressor, INPUT_SIZE, Placement};

        let mut input = vec![0; INPUT_SIZE];
        let mut compressor = Compressor::new(&entry.node, &self.state.options);
        let mut remaining = entry.node.size;
        let mut index_offset = entry.node.index_offset();
        self.output
            .seek(SeekFrom::Start(
                u64::from(entry.next_block) * BLOCK_SIZE as u64,
            ))
            .await?;
        while remaining != 0 {
            let len = remaining.min(input.len() as u64) as usize;
            read_exact(data, &mut input[..len]).await?;
            remaining -= len as u64;
            let mut input = &input[..len];
            while !input.is_empty() {
                let consumed =
                    compressor.encode(input, entry.next_block, self.state.fragments.size)?;
                match compressor.placement {
                    Placement::Inline => entry.node.inline_size = compressor.data.len(),
                    Placement::Fragment => {
                        let mut tail = &compressor.data[..];
                        while !tail.is_empty() {
                            let n = self.state.fragments.append(&mut entry, tail);
                            tail = &tail[n..];
                            if self.state.fragments.used == BLOCK_SIZE {
                                self.flush_fragments().await?;
                            }
                        }
                    }
                    Placement::External => {
                        self.output.write_all(&compressor.data).await?;
                        let blocks = (compressor.data.len() / BLOCK_SIZE) as u32;
                        entry.next_block += blocks; // Worst-case bounds checked before I/O.
                        entry.node.data_block += blocks;
                    }
                }
                for page in self
                    .state
                    .write_indexes(&entry.node, index_offset, &compressor.indexes)
                {
                    write_at(
                        &mut self.output,
                        u64::from(page.block) * BLOCK_SIZE as u64,
                        &page.data,
                    )
                    .await?;
                    self.output
                        .seek(SeekFrom::Start(
                            u64::from(entry.next_block) * BLOCK_SIZE as u64,
                        ))
                        .await?;
                }
                index_offset += compressor.indexes.len() as u64;
                input = &input[consumed..];
            }
        }
        debug_assert_eq!(index_offset, entry.node.index_end());
        let page = self.state.pending_metadata.as_mut().unwrap();
        if compressor.placement == Placement::Inline {
            page.write_at(index_offset, &compressor.data);
        }
        page.write_inode(&entry.node, self.state.nodes.len());
        let header = encode::inode(&entry.node, self.state.nodes.len());
        let map = entry.node.map_header();
        for (offset, bytes) in [
            (
                entry.node.inode_offset(),
                &header[..entry.node.inode_size()],
            ),
            (entry.node.map_offset(), &map[..]),
        ] {
            // write_inode already updated the pending page; patch only headers
            // in pages that were flushed while streaming indexes.
            if offset / BLOCK_SIZE as u64 != u64::from(page.block) {
                write_at(&mut self.output, offset, bytes).await?;
            }
        }
        self.state.commit_payload(entry);
        Ok(())
    }

    async fn prepare_metadata(&mut self, node: &Node) -> Result<()> {
        for page in self.state.prepare_metadata(node) {
            write_at(
                &mut self.output,
                u64::from(page.block) * BLOCK_SIZE as u64,
                &page.data,
            )
            .await?;
        }
        Ok(())
    }

    async fn flush_fragments(&mut self) -> Result<()> {
        let fragments = &mut self.state.fragments;
        if fragments.used != 0 {
            write_at(
                &mut self.output,
                u64::from(fragments.block) * BLOCK_SIZE as u64,
                &fragments.data,
            )
            .await?;
            fragments.used = 0;
        }
        Ok(())
    }

    async fn flush_metadata(&mut self) -> Result<()> {
        if let Some(page) = self.state.pending_metadata.take() {
            write_at(
                &mut self.output,
                u64::from(page.block) * BLOCK_SIZE as u64,
                &page.data,
            )
            .await?;
        }
        Ok(())
    }
}

async fn write_at(
    output: &mut (impl AsyncWrite + AsyncSeek),
    offset: u64,
    bytes: &[u8],
) -> Result<()> {
    output.seek(SeekFrom::Start(offset)).await?;
    output.write_all(bytes).await
}

async fn copy_exact(
    data: &mut impl AsyncRead,
    output: &mut impl AsyncWrite,
    mut size: u64,
) -> Result<()> {
    let mut buf = [0; 8192];
    while size != 0 {
        let len = size.min(buf.len() as u64) as usize;
        let n = read_chunk(data, &mut buf[..len]).await?;
        output.write_all(&buf[..n]).await?;
        size -= n as u64;
    }
    Ok(())
}

async fn read_exact(data: &mut impl AsyncRead, mut buf: &mut [u8]) -> Result<()> {
    while !buf.is_empty() {
        let n = read_chunk(data, buf).await?;
        buf = &mut buf[n..];
    }
    Ok(())
}

// Both payload phases require progress and must reject invalid backend counts.
async fn read_chunk(data: &mut impl AsyncRead, buf: &mut [u8]) -> Result<usize> {
    let n = data.read(buf).await?;
    if n == 0 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "file source shorter than declared size",
        )
        .into());
    }
    if n > buf.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "async reader returned an oversized count",
        )
        .into());
    }
    Ok(n)
}
