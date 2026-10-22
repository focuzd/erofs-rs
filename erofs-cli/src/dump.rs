use crate::source;
use anyhow::Result;
use chrono::{DateTime, Local};
use clap::Args;
use erofs_rs::{
    EroFS,
    r#async::EroFS as AsyncEroFS,
    types::{SB_EXTSLOT_SIZE, SuperBlock},
};
use uuid::Uuid;

#[derive(Args, Debug)]
pub struct DumpArgs {
    path: String,
    /// Additional images in device-table order (repeat for each device).
    #[clap(long = "device", value_name = "PATH_OR_URL")]
    devices: Vec<String>,
}

// Filesystem magic number:                      0xE0F5E1E2
// Filesystem blocksize:                         4096
// Filesystem blocks:                            5136
// Filesystem additional devices:                0
// Filesystem inode metadata start block:        0
// Filesystem shared xattr metadata start block: 0
// Filesystem root nid:                          37
// Filesystem sb_size:                           128
// Filesystem inode count:                       476
// Filesystem created:                           Thu Oct 22 20:33:33 2026
// Filesystem compatible features:               sb_csum mtime plain_xattr_pfx
// Filesystem incompatible features:             chunked_file xattr_prefixes
// Filesystem UUID:                              f980c99a-21e7-4bc2-9f37-cbf3f5cb69c9

struct Feature {
    flag: u32,
    name: &'static str,
}

static COMPAT_FEATURES_LIST: &[Feature] = &[
    Feature {
        flag: 0x01,
        name: "sb_csum",
    },
    Feature {
        flag: 0x02,
        name: "mtime",
    },
    Feature {
        flag: 0x04,
        name: "xattr_filter",
    },
    Feature {
        flag: 0x08,
        name: "shared_ea_in_metabox",
    },
    Feature {
        flag: 0x10,
        name: "plain_xattr_pfx",
    },
    Feature {
        flag: 0x20,
        name: "ishare_xattrs",
    },
];

static INCOMPAT_FEATURES_LIST: &[Feature] = &[
    Feature {
        flag: 0x001,
        name: "lz4_0padding",
    },
    Feature {
        flag: 0x002,
        name: "compr_cfgs",
    },
    Feature {
        flag: 0x002,
        name: "big_pcluster",
    },
    Feature {
        flag: 0x004,
        name: "chunked_file",
    },
    Feature {
        flag: 0x008,
        name: "device_table",
    },
    Feature {
        flag: 0x010,
        name: "ztailpacking",
    },
    Feature {
        flag: 0x020,
        name: "fragments",
    },
    Feature {
        flag: 0x020,
        name: "dedupe",
    },
    Feature {
        flag: 0x040,
        name: "xattr_prefixes",
    },
    Feature {
        flag: 0x080,
        name: "48bit",
    },
    Feature {
        flag: 0x100,
        name: "metabox",
    },
];

fn feature_names(feature_bits: u32, feature_list: &[Feature]) -> Vec<&'static str> {
    feature_list
        .iter()
        .filter(|feature| feature.flag & feature_bits == feature.flag)
        .map(|feature| feature.name)
        .collect()
}

pub async fn dump(args: DumpArgs) -> Result<()> {
    let (block, devices) = if source::is_remote(&args.path) {
        let image = source::http_image(&args.path)?;
        let devices = args
            .devices
            .iter()
            .map(|path| source::http_image(path))
            .collect::<Result<Vec<_>>>()?;
        let fs = AsyncEroFS::new_with_devices(image, devices).await?;
        (*fs.super_block(), fs.devices().to_vec())
    } else {
        let image = source::mmap_image(&args.path)?;
        let devices = args
            .devices
            .iter()
            .map(|path| source::mmap_image(path))
            .collect::<Result<Vec<_>>>()?;
        let fs = EroFS::new_with_devices(image, devices)?;
        (*fs.super_block(), fs.devices().to_vec())
    };

    println!(
        "Filesystem magic number:                      {:#X}",
        block.magic
    );
    println!(
        "Filesystem blocksize:                         {}",
        1 << block.blk_size_bits
    );
    println!(
        "Filesystem blocks:                            {}",
        block.block_count()
    );
    println!(
        "Filesystem additional devices:                {}",
        devices.len()
    );
    for (index, device) in devices.iter().enumerate() {
        println!(
            "Device {}: blocks={}, unified_start={}, tag=b\"{}\"",
            index + 1,
            device.blocks,
            device.unified_start_block,
            device.tag.escape_ascii()
        );
    }
    println!(
        "Filesystem inode metadata start block:        {}",
        block.meta_blk_addr
    );
    println!(
        "Filesystem shared xattr metadata start block: {}",
        block.xattr_blk_addr
    );
    println!(
        "Filesystem root nid:                          {}",
        block.root_inode_id()
    );
    // println!(
    //     "Filesystem lz4_max_distance:                  {}",
    //     block.lz4_max_distance
    // );
    println!(
        "Filesystem sb_size:                           {}",
        SuperBlock::size() + block.ext_slots as usize * SB_EXTSLOT_SIZE
    );
    println!(
        "Filesystem inode count:                       {}",
        block.inos
    );
    let created = block
        .created_unix()
        .and_then(|(seconds, nanos)| DateTime::from_timestamp(seconds, nanos))
        .map(|dt| dt.with_timezone(&Local))
        .and_then(|dt| dt.naive_utc().checked_add_offset(*dt.offset()))
        .map(|dt| dt.format("%a %b %e %H:%M:%S %Y").to_string())
        .unwrap_or_else(|| "<invalid timestamp>".to_string());
    println!("Filesystem created:                           {}", created);
    println!(
        "Filesystem compatible features:               {}",
        feature_names(block.feature_compat, COMPAT_FEATURES_LIST).join(" ")
    );
    println!(
        "Filesystem incompatible features:             {}",
        feature_names(block.feature_incompat, INCOMPAT_FEATURES_LIST).join(" ")
    );

    println!(
        "Filesystem UUID:                              {}",
        Uuid::from_bytes(block.uuid)
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_bits_yield_no_names() {
        assert_eq!(
            feature_names(0x00000000, COMPAT_FEATURES_LIST),
            Vec::<&'static str>::new()
        );
        assert_eq!(
            feature_names(0x00000000, INCOMPAT_FEATURES_LIST),
            Vec::<&'static str>::new()
        );
    }

    #[test]
    fn aliased_bits_yield_every_name() {
        let feature_bits = 0x022;
        let res = vec!["compr_cfgs", "big_pcluster", "fragments", "dedupe"];
        assert_eq!(feature_names(feature_bits, INCOMPAT_FEATURES_LIST), res);
    }

    #[test]
    fn unknown_bits_ignored() {
        assert_eq!(feature_names(0x142, COMPAT_FEATURES_LIST), vec!["mtime"]);
        assert_eq!(
            feature_names(0x300, INCOMPAT_FEATURES_LIST),
            vec!["metabox"]
        );
    }

    #[test]
    fn names_follow_order() {
        let feature_bits = 0x3f;
        let res = vec![
            "sb_csum",
            "mtime",
            "xattr_filter",
            "shared_ea_in_metabox",
            "plain_xattr_pfx",
            "ishare_xattrs",
        ];
        assert_eq!(feature_names(feature_bits, COMPAT_FEATURES_LIST), res);
        let feature_bits = 0x1ff;
        let res = vec![
            "lz4_0padding",
            "compr_cfgs",
            "big_pcluster",
            "chunked_file",
            "device_table",
            "ztailpacking",
            "fragments",
            "dedupe",
            "xattr_prefixes",
            "48bit",
            "metabox",
        ];
        assert_eq!(feature_names(feature_bits, INCOMPAT_FEATURES_LIST), res);
    }
}
