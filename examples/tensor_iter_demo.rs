use slsl::Tensor;

fn main() {
    println!("=== Tensor Iterator Demo ===\n");

    // 1. Basic iterator usage
    println!("1. Basic iterator usage:");
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_vec(data, vec![2, 3]).unwrap();

    println!("   Tensor shape: {:?}", tensor.dims());
    println!("   Using tensor.iter():");
    for elem in tensor.iter() {
        let ptr = unsafe { elem.as_ptr(tensor.as_ptr()) };
        println!(
            "     Indices: {:?}, Pointer: {:p}",
            elem.indices.as_slice(),
            ptr
        );
    }

    // 2. For loop syntax
    println!("\n2. For loop syntax (for _ in &tensor):");
    for elem in &tensor {
        let ptr = unsafe { elem.as_ptr(tensor.as_ptr()) };
        println!(
            "   Indices: {:?}, Pointer: {:p}",
            elem.indices.as_slice(),
            ptr
        );
    }

    // 3. 1D tensor iteration
    println!("\n3. 1D tensor iteration:");
    let tensor_1d = Tensor::from_vec(vec![10.0f32, 20.0, 30.0], vec![3]).unwrap();
    for elem in &tensor_1d {
        println!("   Index: {:?}", elem.indices.as_slice());
    }

    // 4. 3D tensor iteration
    println!("\n4. 3D tensor iteration:");
    let data_3d: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let tensor_3d = Tensor::from_vec(data_3d, vec![2, 2, 2]).unwrap();

    println!("   3D tensor shape: {:?}", tensor_3d.dims());

    for (count, elem) in (&tensor_3d).into_iter().enumerate() {
        println!(
            "   Element {}: indices {:?}",
            count,
            elem.indices.as_slice()
        );
    }

    // 5. Iterator methods
    println!("\n5. Iterator methods:");
    let iter = tensor.iter();
    println!("   Length: {}", iter.len());
    println!("   Is empty: {}", iter.is_empty());

    // 6. Collecting to vector
    println!("\n6. Collecting to vector:");
    let indices: Vec<_> = tensor.iter().map(|elem| elem.indices).collect();
    println!(
        "   Collected indices: {:?}",
        indices.iter().map(|idx| idx.as_slice()).collect::<Vec<_>>()
    );

    // 7. Conditional filtering
    println!("\n7. Conditional filtering:");
    let filtered: Vec<_> = tensor
        .iter()
        .filter(|elem| elem.indices[0] == 0) // Only take first row
        .map(|elem| elem.indices)
        .collect();
    println!(
        "   First row indices: {:?}",
        filtered
            .iter()
            .map(|idx| idx.as_slice())
            .collect::<Vec<_>>()
    );

    println!("\n=== Demo completed ===");
}
