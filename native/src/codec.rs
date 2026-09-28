//! Lossless float32 bit-pattern codec, independent of engine and batch semantics.

use std::slice;

const MAX_VALUES: usize = 1_000_000;

/// Requires raw-size capacity even when RLE wins. Returns bytes written, -1 for
/// invalid pointers/count, or -2 for insufficient capacity; errors write nothing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mirror_encode_f32_runs(
    values: *const f32, count: usize, out: *mut u8, capacity: usize,
) -> isize {
    if values.is_null() || out.is_null() || count > MAX_VALUES { return -1; }
    let raw_length = count * 4 + 1;
    if capacity < raw_length { return -2; }
    let input = unsafe { slice::from_raw_parts(values.cast::<u32>(), count) };
    let output = unsafe { slice::from_raw_parts_mut(out, raw_length) };
    let (mut read, mut write) = (0, 1);
    output[0] = 1;
    while read < count {
        let start = read;
        let bits = input[read];
        read += 1;
        while read < count && input[read] == bits { read += 1; }
        if write + 8 >= raw_length {
            output[0] = 0;
            for (value, bytes) in input.iter().zip(output[1..].chunks_exact_mut(4)) {
                bytes.copy_from_slice(&value.to_le_bytes());
            }
            return raw_length as isize;
        }
        output[write..write + 4].copy_from_slice(&((read - start) as u32).to_le_bytes());
        output[write + 4..write + 8].copy_from_slice(&bits.to_le_bytes());
        write += 8;
    }
    write as isize
}

/// Returns -1 for invalid arguments, -2 for malformed format/length, -3 for
/// invalid run totals. Validate the complete stream before touching output.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mirror_decode_f32_runs(
    encoded: *const u8, length: usize, out: *mut f32, count: usize,
) -> i32 {
    if encoded.is_null() || out.is_null() || length == 0 || count > MAX_VALUES
        || length > MAX_VALUES * 8 + 1 { return -1; }
    let input = unsafe { slice::from_raw_parts(encoded, length) };
    if input[0] == 0 {
        if length != count * 4 + 1 { return -2; }
        let output = unsafe { slice::from_raw_parts_mut(out.cast::<u32>(), count) };
        for (value, bytes) in output.iter_mut().zip(input[1..].chunks_exact(4)) {
            *value = u32::from_le_bytes(bytes.try_into().unwrap());
        }
        return 0;
    }
    if input[0] != 1 || (length - 1) % 8 != 0 { return -2; }
    let mut position = 0;
    for bytes in input[1..].chunks_exact(8) {
        let run = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
        if run == 0 || run > count - position { return -3; }
        position += run;
    }
    if position != count { return -3; }
    let output = unsafe { slice::from_raw_parts_mut(out.cast::<u32>(), count) };
    position = 0;
    for bytes in input[1..].chunks_exact(8) {
        let run = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
        let value = u32::from_le_bytes(bytes[4..].try_into().unwrap());
        output[position..position + run].fill(value);
        position += run;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::{null, null_mut};

    fn encode(bits: &[u32]) -> Vec<u8> {
        let values: Vec<f32> = bits.iter().copied().map(f32::from_bits).collect();
        let mut out = vec![0xa5; bits.len() * 4 + 1];
        let length = unsafe {
            mirror_encode_f32_runs(values.as_ptr(), values.len(), out.as_mut_ptr(), out.len())
        };
        assert!(length > 0);
        out.truncate(length as usize);
        out
    }

    fn decode(bytes: &[u8], count: usize) -> Vec<u32> {
        let mut out = vec![0.0f32; count];
        assert_eq!(unsafe {
            mirror_decode_f32_runs(bytes.as_ptr(), bytes.len(), out.as_mut_ptr(), count)
        }, 0);
        out.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    fn raw_encoding_is_little_endian_and_preserves_special_float_bits() {
        let bits = [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00001, 0x7fa12345, 0xffc12345];
        let bytes = encode(&bits);
        assert_eq!(bytes, [
            0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 128, 127, 0, 0, 128, 255,
            1, 0, 192, 127, 69, 35, 161, 127, 69, 35, 193, 255,
        ]);
        assert_eq!(decode(&bytes, bits.len()), bits);
    }

    #[test]
    fn rle_encoding_compares_bits_including_nan_payload_and_signed_zero() {
        let bits = [vec![0x7fa12345; 10], vec![0x80000000; 6], vec![0; 5]].concat();
        let bytes = encode(&bits);
        assert_eq!(bytes, [
            1, 10, 0, 0, 0, 69, 35, 161, 127,
            6, 0, 0, 0, 0, 0, 0, 128,
            5, 0, 0, 0, 0, 0, 0, 0,
        ]);
        assert_eq!(decode(&bytes, bits.len()), bits);
    }

    #[test]
    fn raw_fallback_wins_ties_and_incompressible_runs() {
        assert_eq!(encode(&[0x3f800000; 2]), [0, 0, 0, 128, 63, 0, 0, 128, 63]);
        assert_eq!(encode(&[0x3f800000; 3]), [1, 3, 0, 0, 0, 0, 0, 128, 63]);
        let bits = [vec![42; 20], (0..30).collect()].concat();
        let bytes = encode(&bits);
        assert_eq!(bytes[0], 0);
        assert_eq!(bytes.len(), bits.len() * 4 + 1);
        assert_eq!(decode(&bytes, bits.len()), bits);
    }

    #[test]
    fn empty_encoding_and_maximum_allowed_count_round_trip() {
        assert_eq!(encode(&[]), [1]);
        assert!(decode(&[1], 0).is_empty());
        assert!(decode(&[0], 0).is_empty());
        let bits = vec![0xffc12345; 1_000_000];
        let bytes = encode(&bits);
        assert_eq!(bytes.len(), 9);
        assert_eq!(decode(&bytes, bits.len()), bits);
    }

    #[test]
    fn codec_rejects_invalid_pointers_counts_capacity_without_writes() {
        let values = [f32::from_bits(0x7fa12345); 3];
        let mut encoded = [0xa5; 13];
        let mut decoded = [f32::from_bits(0x12345678); 3];
        unsafe {
            assert_eq!(mirror_encode_f32_runs(null(), 3, encoded.as_mut_ptr(), 13), -1);
            assert_eq!(mirror_encode_f32_runs(values.as_ptr(), 3, null_mut(), 13), -1);
            assert_eq!(mirror_encode_f32_runs(values.as_ptr(), 1_000_001, encoded.as_mut_ptr(), 13), -1);
            assert_eq!(mirror_encode_f32_runs(values.as_ptr(), usize::MAX, encoded.as_mut_ptr(), 13), -1);
            assert_eq!(mirror_encode_f32_runs(values.as_ptr(), 3, encoded.as_mut_ptr(), 12), -2);
            assert_eq!(mirror_encode_f32_runs(values.as_ptr(), 0, encoded.as_mut_ptr(), 0), -2);
            assert_eq!(mirror_decode_f32_runs(null(), 1, decoded.as_mut_ptr(), 3), -1);
            assert_eq!(mirror_decode_f32_runs(encoded.as_ptr(), 1, null_mut(), 3), -1);
            assert_eq!(mirror_decode_f32_runs(encoded.as_ptr(), 0, decoded.as_mut_ptr(), 3), -1);
            assert_eq!(mirror_decode_f32_runs(encoded.as_ptr(), 1, decoded.as_mut_ptr(), 1_000_001), -1);
            assert_eq!(mirror_decode_f32_runs(encoded.as_ptr(), usize::MAX, decoded.as_mut_ptr(), 3), -1);
        }
        assert_eq!(encoded, [0xa5; 13]);
        assert_eq!(decoded.map(f32::to_bits), [0x12345678; 3]);
    }

    #[test]
    fn malformed_decode_is_atomic_including_late_bad_runs() {
        for (bytes, count, status) in [
            (vec![2], 3, -2),
            (vec![0, 0, 0, 0], 1, -2),
            (vec![0, 0, 0, 0, 0, 0], 1, -2),
            (vec![1, 1], 3, -2),
            (vec![1, 0, 0, 0, 0, 42, 0, 0, 0], 3, -3),
            (vec![1, 4, 0, 0, 0, 42, 0, 0, 0], 3, -3),
            (vec![1, 2, 0, 0, 0, 42, 0, 0, 0], 3, -3),
            (vec![1], 3, -3),
            (vec![1, 1, 0, 0, 0, 42, 0, 0, 0, 0, 0, 0, 0, 43, 0, 0, 0], 3, -3),
            (vec![1, 1, 0, 0, 0, 42, 0, 0, 0, 255, 255, 255, 255, 43, 0, 0, 0], 3, -3),
        ] {
            let mut out = [f32::from_bits(0x12345678); 3];
            assert_eq!(unsafe {
                mirror_decode_f32_runs(bytes.as_ptr(), bytes.len(), out.as_mut_ptr(), count)
            }, status, "{bytes:?}");
            assert_eq!(out.map(f32::to_bits), [0x12345678; 3], "{bytes:?}");
        }
    }
}
