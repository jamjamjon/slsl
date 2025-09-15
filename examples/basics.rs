use slsl::{s, Tensor};

fn main() -> anyhow::Result<()> {
    // creation
    let tensor = Tensor::rand(1., 10., [5, 5, 5])?;
    println!("# initial: {tensor:?}",);

    // iter_dim
    for elem in tensor.iter_dim(0) {
        println!("# elem: {elem:?}");
    }

    // slice
    let slice = tensor.slice(s![None, 1..5, -4.., ..]);
    println!("# slice: {slice:?}",);

    // squeeze_all
    let squeezed = slice.squeeze(0)?;
    println!("# squeezed: {squeezed:?}",);

    // unsqueeze
    let unsqueezed = squeezed.unsqueeze(0)?;
    println!("# unsqueezed: {unsqueezed:?}",);

    // sum
    let sum = unsqueezed.sum(-1)?;
    println!("# sum: {sum:?}",);

    // min_max_argmin_argmax
    let (min, max, argmin, argmax) = sum.min_max_argmin_argmax(-1)?;
    println!("# min: {min:?}",);
    println!("# max: {max:?}",);
    println!("# argmin: {argmin:?}",);
    println!("# argmax: {argmax:?}",);

    // to vec3
    let vec3 = sum.to_vec3::<f64>()?;
    println!("# vec3: {vec3:?}",);

    // to vec2
    let vec2 = sum.squeeze(0)?.to_vec2::<f64>()?;
    println!("vec2: {vec2:?}",);

    // to flat vec
    let flat = sum.to_flat_vec::<f64>()?;
    println!("# flat vec: {flat:?}",);

    Ok(())
}
