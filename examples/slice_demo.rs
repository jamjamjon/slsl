use slsl::{s, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 SLSL Tensor Slicing Demo - Complete NumPy Style Slicing System");
    println!("{}", "=".repeat(70));

    // Create test tensors with different dimensions
    let data_1d: Vec<f32> = (0..10).map(|x| x as f32).collect();
    let tensor_1d = Tensor::from_vec(data_1d, [10])?;
    println!("\n📊 1D Tensor: shape={:?}", tensor_1d.shape().as_slice());
    println!("Data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]");

    let data_2d: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let tensor_2d = Tensor::from_vec(data_2d, [4, 6])?;
    println!("\n📊 2D Tensor: shape={:?}", tensor_2d.shape().as_slice());
    println!("Data: 4x6 matrix with values 0-23");

    let data_3d: Vec<f32> = (0..60).map(|x| x as f32).collect();
    let tensor_3d = Tensor::from_vec(data_3d, [3, 4, 5])?;
    println!("\n📊 3D Tensor: shape={:?}", tensor_3d.shape().as_slice());
    println!("Data: 3x4x5 tensor with values 0-59");

    println!("\n{}", "=".repeat(70));
    println!("🔍 SLICE OPERATIONS SHOWCASE");
    println!("{}", "=".repeat(70));

    // ========== Index Operations ==========
    println!("\n🎯 INDEX OPERATIONS (Dimension Reduction Access)");
    println!("{}", "-".repeat(50));

    // s![2] - 2nd element (dimension reduction)
    let result = tensor_1d.slice(s![2]);
    println!("¥¥¥ {result:?}");
    println!(
        "s![2] -> 2nd element: shape={:?}",
        result.shape().as_slice()
    );

    // s![-1] - last element
    let result = tensor_1d.slice(s![-1]);
    println!(
        "s![-1] -> last element: shape={:?}",
        result.shape().as_slice()
    );

    // s![-2] - second to last element
    let result = tensor_1d.slice(s![-2]);
    println!(
        "s![-2] -> 2nd to last element: shape={:?}",
        result.shape().as_slice()
    );

    // 2D tensor indexing
    let result = tensor_2d.slice(s![1, 3]);
    println!(
        "s![1, 3] -> 2D indexing: shape={:?}",
        result.shape().as_slice()
    );

    let result = tensor_2d.slice(s![-1, -1]);
    println!(
        "s![-1, -1] -> 2D negative indexing: shape={:?}",
        result.shape().as_slice()
    );

    // ========== Range Operations ==========
    println!("\n🎯 RANGE OPERATIONS (Range Slicing)");
    println!("{}", "-".repeat(50));

    // s![1..5] - from 1 to 5 (exclusive)
    let result = tensor_1d.slice(s![1..5]);
    println!(
        "s![1..5] -> half-open interval: shape={:?}",
        result.shape().as_slice()
    );

    // s![1..=5] - from 1 to 5 (inclusive)
    let result = tensor_1d.slice(s![1..=5]);
    println!(
        "s![1..=5] -> closed interval: shape={:?}",
        result.shape().as_slice()
    );

    // s![..] - all elements
    let result = tensor_1d.slice(s![..]);
    println!(
        "s![..] -> all elements: shape={:?}",
        result.shape().as_slice()
    );

    // s![2..] - from 2 to end
    let result = tensor_1d.slice(s![2..]);
    println!(
        "s![2..] -> from 2 to end: shape={:?}",
        result.shape().as_slice()
    );

    // s![..7] - from start to 7 (exclusive)
    let result = tensor_1d.slice(s![..7]);
    println!(
        "s![..7] -> from start to 7: shape={:?}",
        result.shape().as_slice()
    );

    // s![..=7] - from start to 7 (inclusive)
    let result = tensor_1d.slice(s![..=7]);
    println!(
        "s![..=7] -> from start to 7 (inclusive): shape={:?}",
        result.shape().as_slice()
    );

    // ========== Negative Range Operations ==========
    println!("\n🎯 NEGATIVE RANGE OPERATIONS");
    println!("{}", "-".repeat(50));

    // s![-5..5] - negative start to positive end
    let result = tensor_1d.slice(s![-5..5]);
    println!(
        "s![-5..5] -> negative start to positive end: shape={:?}",
        result.shape().as_slice()
    );

    // s![1..8] - positive start to positive end
    let result = tensor_1d.slice(s![1..8]);
    println!(
        "s![1..8] -> positive range: shape={:?}",
        result.shape().as_slice()
    );

    // s![-5..-2] - negative start to negative end
    let result = tensor_1d.slice(s![-5..-2]);
    println!(
        "s![-5..-2] -> negative start to negative end: shape={:?}",
        result.shape().as_slice()
    );

    // s![-2..] - negative start to end
    let result = tensor_1d.slice(s![-2..]);
    println!(
        "s![-2..] -> negative start to end: shape={:?}",
        result.shape().as_slice()
    );

    // s![..-2] - start to negative end
    let result = tensor_1d.slice(s![..-2]);
    println!(
        "s![..-2] -> start to negative end: shape={:?}",
        result.shape().as_slice()
    );

    // ========== Step Operations ==========
    println!("\n🎯 STEP OPERATIONS (Stride Slicing)");
    println!("{}", "-".repeat(50));

    // s![2..;3] - from 2 to end, step 3
    let result = tensor_1d.slice(s![2..;3]);
    println!(
        "s![2..;3] -> from 2 with step 3: shape={:?}",
        result.shape().as_slice()
    );

    // s![..=8;2] - from start to 8 (inclusive), step 2
    let result = tensor_1d.slice(s![..=8;2]);
    println!(
        "s![..=8;2] -> to 8 with step 2: shape={:?}",
        result.shape().as_slice()
    );

    // s![1..9;2] - from 1 to 9, step 2
    let result = tensor_1d.slice(s![1..9;2]);
    println!(
        "s![1..9;2] -> 1 to 9 with step 2: shape={:?}",
        result.shape().as_slice()
    );

    // s![..;2] - all elements, step 2
    let result = tensor_1d.slice(s![..;2]);
    println!(
        "s![..;2] -> all with step 2: shape={:?}",
        result.shape().as_slice()
    );

    // ========== NewAxis Operations ==========
    println!("\n🎯 NEWAXIS OPERATIONS (Dimension Expansion)");
    println!("{}", "-".repeat(50));

    // s![None] - add new axis
    let result = tensor_1d.slice(s![None]);
    println!(
        "s![None] -> add new axis: shape={:?}",
        result.shape().as_slice()
    );

    // s![0, None, 1] - mixed indexing and new axis
    let result = tensor_2d.slice(s![0, None]);
    println!(
        "s![0, None] -> index + new axis: shape={:?}",
        result.shape().as_slice()
    );

    // s![None, ..] - new axis at beginning
    let result = tensor_1d.slice(s![None, ..]);
    println!(
        "s![None, ..] -> new axis at start: shape={:?}",
        result.shape().as_slice()
    );

    // ========== Multi-dimensional Operations ==========
    println!("\n🎯 MULTI-DIMENSIONAL OPERATIONS");
    println!("{}", "-".repeat(50));

    // 2D slicing
    let result = tensor_2d.slice(s![1..3, 2..5]);
    println!(
        "s![1..3, 2..5] -> 2D range slice: shape={:?}",
        result.shape().as_slice()
    );

    // Mixed 2D operations
    let result = tensor_2d.slice(s![0, ..]);
    println!(
        "s![0, ..] -> first row: shape={:?}",
        result.shape().as_slice()
    );

    let result = tensor_2d.slice(s![.., 0]);
    println!(
        "s![.., 0] -> first column: shape={:?}",
        result.shape().as_slice()
    );

    // 3D slicing
    let result = tensor_3d.slice(s![1, .., 2..4]);
    println!(
        "s![1, .., 2..4] -> 3D mixed slice: shape={:?}",
        result.shape().as_slice()
    );

    let result = tensor_3d.slice(s![..2, 1..3, ..]);
    println!(
        "s![..2, 1..3, ..] -> 3D range slice: shape={:?}",
        result.shape().as_slice()
    );

    // ========== Complex Mixed Operations ==========
    println!("\n🎯 COMPLEX MIXED OPERATIONS");
    println!("{}", "-".repeat(50));

    // Step with ranges in 2D
    let result = tensor_2d.slice(s![..;2, 1..5]);
    println!(
        "s![..;2, 1..5] -> 2D step slice: shape={:?}",
        result.shape().as_slice()
    );

    // Mixed indexing, ranges, and steps
    let result = tensor_3d.slice(s![0, 1.., ..3]);
    println!(
        "s![0, 1.., ..3] -> mixed operations: shape={:?}",
        result.shape().as_slice()
    );

    // Negative indices with steps
    let result = tensor_2d.slice(s![-3.., ..-1]);
    println!(
        "s![-3.., ..-1] -> negative range: shape={:?}",
        result.shape().as_slice()
    );

    // ========== Chain Slicing Operations ==========
    println!("\n🎯 CHAIN SLICING OPERATIONS (View Chaining)");
    println!("{}", "-".repeat(50));

    // Chain slicing with TensorView
    let view1 = tensor_3d.slice(s![1, ..]);
    let view2 = view1.slice(s![1..3]);
    println!(
        "tensor_3d.slice(s![1, ..]).slice(s![1..3]) -> chained view: shape={:?}",
        view2.shape().as_slice()
    );

    // Multiple chain operations
    let view3 = tensor_3d.slice(s![..2]).slice(s![1]).slice(s![2..]);
    println!(
        "tensor_3d.slice(s![..2]).slice(s![1]).slice(s![2..]) -> triple chain: shape={:?}",
        view3.shape().as_slice()
    );

    // ========== Note: Mutable Slicing Operations ==========
    println!("\n🎯 NOTE: MUTABLE SLICING OPERATIONS");
    println!("{}", "-".repeat(50));
    println!("⚠️  Mutable slicing operations have been removed for memory safety.");
    println!("📝 Use immutable slice() method for all slicing operations.");
    println!("🔒 This design ensures zero-copy safety and prevents data races.");

    // Create a tensor for demonstration
    let data_demo: Vec<f32> = (0..20).map(|x| x as f32).collect();
    let tensor_demo = Tensor::from_vec(data_demo, [4, 5])?;

    // Immutable slice operations (recommended approach)
    let view = tensor_demo.slice(s![1..3, ..]);
    println!(
        "tensor_demo.slice(s![1..3, ..]) -> immutable view: shape={:?}",
        view.shape().as_slice()
    );

    // Chain immutable slicing (now works perfectly!)
    let view2 = tensor_demo.slice(s![..]).slice(s![1..3]);
    println!(
        "tensor_demo.slice(s![..]).slice(s![1..3]) -> chained immutable: shape={:?}",
        view2.shape().as_slice()
    );

    // Multiple chain operations
    let chained_view = tensor_demo.slice(s![1..]).slice(s![..2]);
    println!(
        "tensor_demo.slice(s![1..]).slice(s![..2]) -> multi-chain: shape={:?}",
        chained_view.shape().as_slice()
    );

    // ========== Performance Showcase ==========
    println!("\n🎯 PERFORMANCE SHOWCASE (Fast Paths)");
    println!("{}", "-".repeat(50));

    // Single index fast path
    let fast_single = tensor_2d.slice(s![2]);
    println!(
        "s![2] -> single index fast path: shape={:?}",
        fast_single.shape().as_slice()
    );

    // Double index fast path
    let fast_double = tensor_2d.slice(s![1, 3]);
    println!(
        "s![1, 3] -> double index fast path: shape={:?}",
        fast_double.shape().as_slice()
    );

    // 2D range fast path
    let fast_2d_range = tensor_2d.slice(s![1..3, 2..5]);
    println!(
        "s![1..3, 2..5] -> 2D range fast path: shape={:?}",
        fast_2d_range.shape().as_slice()
    );

    println!("\n{}", "=".repeat(70));
    println!("✅ All slice operations completed successfully!");
    println!("🚀 SLSL tensor slicing provides zero-copy, high-performance operations");
    println!("📈 Fast paths automatically optimize common patterns");
    println!("🔗 Chain operations enable complex indexing workflows");
    println!("{}", "=".repeat(70));

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use slsl::SliceElem;

    #[test]
    fn test_basic_indexing() {
        let data: Vec<f32> = (0..10).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(data, [10]).unwrap();

        let result = tensor.slice(s![2]);
        assert_eq!(result.shape().as_slice(), assert_eq!(result.shape().as_slice(), &[])[] as assert_eq!(result.shape().as_slice(), &[])[usize]);
    }

    #[test]
    fn test_negative_indexing() {
        let data: Vec<f32> = (0..10).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(data, [10]).unwrap();

        let result = tensor.slice(s![-1]);
        assert_eq!(result.shape().as_slice(), assert_eq!(result.shape().as_slice(), &[])[] as assert_eq!(result.shape().as_slice(), &[])[usize]);
    }

    #[test]
    fn test_range_slicing() {
        let data: Vec<f32> = (0..10).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(data, [10]).unwrap();

        let result = tensor.slice(s![1..5]);
        assert_eq!(result.shape().as_slice(), &[4]);

        let result = tensor.slice(s![..]);
        assert_eq!(result.shape().as_slice(), &[10]);
    }

    #[test]
    fn test_inclusive_range() {
        let data: Vec<f32> = (0..10).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(data, [10]).unwrap();

        let result = tensor.slice(s![1..=5]);
        assert_eq!(result.shape().as_slice(), &[5]);
    }

    #[test]
    fn test_step_slicing() {
        let data: Vec<f32> = (0..10).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(data, [10]).unwrap();

        let result = tensor.slice(s![..;2]);
        assert_eq!(result.shape().as_slice(), &[5]);
    }

    #[test]
    fn test_new_axis() {
        let data: Vec<f32> = (0..10).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(data, [10]).unwrap();

        let result = tensor.slice(s![None]);
        assert_eq!(result.shape().as_slice(), &[1, 10]);
    }

    #[test]
    fn test_chain_slicing() {
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(data, [4, 6]).unwrap();

        let view1 = tensor.slice(s![1..3]);
        let view2 = view1.slice(s![.., 2..5]);
        assert_eq!(view2.shape().as_slice(), &[2, 3]);
    }

    #[test]
    fn test_immutable_slicing_chain() {
        let data: Vec<f32> = (0..20).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(data, [4, 5]).unwrap();

        let view = tensor.slice(s![1..3]);
        assert_eq!(view.shape().as_slice(), &[2, 5]);

        let chained_view = tensor.slice(s![..]).slice(s![1..3]);
        assert_eq!(chained_view.shape().as_slice(), &[2, 5]);
    }
}
