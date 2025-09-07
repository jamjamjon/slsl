use crate::{DType, Tensor, TensorElement, TensorView};
use half::{bf16, f16};

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rank = self.rank();
        let desc = match rank {
            0 => "Tensor(Scalar)".to_string(),
            r => format!("Tensor({r}D)"),
        };
        writeln!(f, "{desc} {{")?;
        write!(f, "  data: ")?;
        self.format_data(f)?;
        writeln!(
            f,
            "  dtype: {:?}, shape: {:?}",
            self.dtype(),
            self.shape().as_slice()
        )?;
        writeln!(f, "}}")
    }
}

impl std::fmt::Debug for TensorView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rank = self.rank();
        let desc = match rank {
            0 => "TensorView(Scalar)".to_string(),
            r => format!("TensorView({r}D)"),
        };
        writeln!(f, "{desc} {{")?;
        write!(f, "  data: ")?;
        self.format_data(f)?;
        writeln!(
            f,
            "  dtype: {:?}, shape: {:?}",
            self.dtype(),
            self.shape().as_slice()
        )?;
        writeln!(f, "}}")
    }
}

/// Parameters for formatting tensor data
struct FormatParams<'a> {
    shape: &'a [usize],
    truncate: bool,
    max_per_dim: usize,
    line_width: usize,
    current_line_length: usize,
}

impl Tensor {
    /// Format tensor data for debug output with intelligent truncation
    fn format_data(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let _shape = self.shape().as_slice();
        let total_elements = self.numel();

        // Threshold for showing all elements vs truncating
        const SMALL_TENSOR_THRESHOLD: usize = 100;
        const MAX_ELEMENTS_PER_DIM: usize = 6;

        if total_elements == 0 {
            return write!(f, "[]");
        }

        if total_elements <= SMALL_TENSOR_THRESHOLD {
            // Show all elements for small tensors
            self.format_all_data(f)
        } else {
            // Show truncated view for large tensors
            self.format_truncated_data(f, MAX_ELEMENTS_PER_DIM)
        }
    }

    /// Get terminal width for better formatting (fallback to 80 if unavailable)
    fn get_terminal_width(&self) -> usize {
        // Simple fallback without external dependencies
        80
    }

    /// Format all tensor data (for small tensors)
    fn format_all_data(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.dtype() {
            DType::Bool => self.format_typed_data::<bool>(f, false, 0),
            DType::Int8 => self.format_typed_data::<i8>(f, false, 0),
            DType::Int16 => self.format_typed_data::<i16>(f, false, 0),
            DType::Int32 => self.format_typed_data::<i32>(f, false, 0),
            DType::Int64 => self.format_typed_data::<i64>(f, false, 0),
            DType::Uint8 => self.format_typed_data::<u8>(f, false, 0),
            DType::Uint16 => self.format_typed_data::<u16>(f, false, 0),
            DType::Uint32 => self.format_typed_data::<u32>(f, false, 0),
            DType::Uint64 => self.format_typed_data::<u64>(f, false, 0),
            DType::Fp16 => self.format_typed_data::<f16>(f, false, 0),
            DType::Fp32 => self.format_typed_data::<f32>(f, false, 0),
            DType::Fp64 => self.format_typed_data::<f64>(f, false, 0),
            DType::Bf16 => self.format_typed_data::<bf16>(f, false, 0),
            _ => Err(std::fmt::Error),
        }
    }

    /// Format truncated tensor data (for large tensors)
    fn format_truncated_data(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        max_per_dim: usize,
    ) -> std::fmt::Result {
        match self.dtype() {
            DType::Bool => self.format_typed_data::<bool>(f, true, max_per_dim),
            DType::Int8 => self.format_typed_data::<i8>(f, true, max_per_dim),
            DType::Int16 => self.format_typed_data::<i16>(f, true, max_per_dim),
            DType::Int32 => self.format_typed_data::<i32>(f, true, max_per_dim),
            DType::Int64 => self.format_typed_data::<i64>(f, true, max_per_dim),
            DType::Uint8 => self.format_typed_data::<u8>(f, true, max_per_dim),
            DType::Uint16 => self.format_typed_data::<u16>(f, true, max_per_dim),
            DType::Uint32 => self.format_typed_data::<u32>(f, true, max_per_dim),
            DType::Uint64 => self.format_typed_data::<u64>(f, true, max_per_dim),
            DType::Fp16 => self.format_typed_data::<f16>(f, true, max_per_dim),
            DType::Fp32 => self.format_typed_data::<f32>(f, true, max_per_dim),
            DType::Fp64 => self.format_typed_data::<f64>(f, true, max_per_dim),
            DType::Bf16 => self.format_typed_data::<bf16>(f, true, max_per_dim),
            _ => Err(std::fmt::Error),
        }
    }

    /// Format typed tensor data with optional truncation
    fn format_typed_data<T: TensorElement + std::fmt::Display>(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        truncate: bool,
        max_per_dim: usize,
    ) -> std::fmt::Result {
        let shape = self.shape().as_slice();

        if shape.is_empty() {
            // Scalar tensor
            let value = self.at::<T>(crate::Shape::empty());
            return self.format_single_value(f, &value);
        }

        let terminal_width = self.get_terminal_width();
        let params = FormatParams {
            shape,
            truncate,
            max_per_dim,
            line_width: terminal_width,
            current_line_length: 9, // Initial indentation for "  data: ["
        };

        self.format_recursive::<T>(f, &params, &[], 0)
    }

    /// Recursively format multi-dimensional tensor data with line width control
    fn format_recursive<T: TensorElement + std::fmt::Display>(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        params: &FormatParams,
        indices: &[usize],
        depth: usize,
    ) -> std::fmt::Result {
        let FormatParams {
            shape,
            truncate,
            max_per_dim,
            line_width,
            current_line_length,
        } = params;

        if shape.is_empty() {
            return Ok(());
        }

        let current_dim = shape[0];
        let remaining_shape = &shape[1..];
        let is_1d = remaining_shape.is_empty();
        let is_2d = remaining_shape.len() == 1;
        let is_outermost = depth == 0;
        let total_dims = shape.len() + depth;

        write!(f, "[")?;
        let mut current_line_len = *current_line_length + 1;

        let (show_count, need_ellipsis) = if *truncate && current_dim > *max_per_dim {
            (*max_per_dim / 2, true)
        } else {
            (current_dim, false)
        };

        // Show first elements
        for i in 0..show_count.min(current_dim) {
            if i > 0 {
                if is_1d {
                    // 1D tensor: check line width for smart wrapping
                    if current_line_len > line_width.saturating_sub(20) {
                        write!(f, ",\n         ")?; // 9 spaces to align with "  data: ["
                        current_line_len = 9;
                    } else {
                        write!(f, ", ")?;
                        current_line_len += 2;
                    }
                } else if is_outermost {
                    // Multi-dimensional outermost: newline with proper indentation
                    write!(f, ",\n         ")?; // 9 spaces to align with "  data: ["
                    current_line_len = 9;
                } else if total_dims >= 3 && is_2d {
                    // For 3D+ tensors, put each 2D slice on new line with extra indentation
                    let indent = " ".repeat(9 + depth * 2 - 1);
                    write!(f, ",\n{indent}")?;
                    current_line_len = 9 + depth * 2;
                } else {
                    // Inner dimensions: align to the position of the innermost '['
                    let indent = " ".repeat(9 + depth + 1);
                    write!(f, ",\n{indent}")?;
                    current_line_len = 9 + depth + 1;
                }
            }

            let mut new_indices = indices.to_vec();
            new_indices.push(i);

            if remaining_shape.is_empty() {
                // Leaf element
                let indices_array = crate::Shape::from_slice(&new_indices);
                let value = self.at::<T>(indices_array);
                let value_str = self.format_value_to_string(&value);
                current_line_len += value_str.len();
                self.format_single_value(f, &value)?;
            } else {
                // Recursive call for sub-tensor
                let new_params = FormatParams {
                    shape: remaining_shape,
                    truncate: *truncate,
                    max_per_dim: *max_per_dim,
                    line_width: *line_width,
                    current_line_length: current_line_len,
                };
                self.format_recursive::<T>(f, &new_params, &new_indices, depth + 1)?;
            }
        }

        // Show ellipsis if truncated
        if need_ellipsis && current_dim > *max_per_dim {
            if show_count > 0 {
                if is_1d {
                    if current_line_len > line_width.saturating_sub(20) {
                        write!(f, ",\n         ")?;
                        current_line_len = 9;
                    } else {
                        write!(f, ", ")?;
                        current_line_len += 2;
                    }
                } else if is_outermost {
                    write!(f, ",\n         ")?;
                    current_line_len = 9;
                } else if total_dims >= 3 && is_2d {
                    let indent = " ".repeat(9 + depth * 2);
                    write!(f, ",\n{indent}")?;
                    current_line_len = 9 + depth * 2;
                } else {
                    let indent = " ".repeat(9 + depth + 1);
                    write!(f, ",\n{indent}")?;
                    current_line_len = 9 + depth + 1;
                }
            }
            write!(f, "...")?;
            current_line_len += 3;

            // Show last elements
            let last_start = current_dim - (*max_per_dim - show_count);
            for i in last_start..current_dim {
                if is_1d {
                    if current_line_len > line_width.saturating_sub(20) {
                        write!(f, ",\n         ")?;
                        current_line_len = 9;
                    } else {
                        write!(f, ", ")?;
                        current_line_len += 2;
                    }
                } else if is_outermost {
                    write!(f, ",\n         ")?;
                    current_line_len = 9;
                } else if total_dims >= 3 && is_2d {
                    let indent = " ".repeat(9 + depth * 2);
                    write!(f, ",\n{indent}")?;
                    current_line_len = 9 + depth * 2;
                } else {
                    let indent = " ".repeat(9 + depth + 1);
                    write!(f, ",\n{indent}")?;
                    current_line_len = 9 + depth + 1;
                }

                let mut new_indices = indices.to_vec();
                new_indices.push(i);

                if remaining_shape.is_empty() {
                    // Leaf element
                    let indices_array = crate::Shape::from_slice(&new_indices);
                    let value = self.at::<T>(indices_array);
                    let value_str = self.format_value_to_string(&value);
                    current_line_len += value_str.len();
                    self.format_single_value(f, &value)?;
                } else {
                    // Recursive call for sub-tensor
                    let new_params = FormatParams {
                        shape: remaining_shape,
                        truncate: *truncate,
                        max_per_dim: *max_per_dim,
                        line_width: *line_width,
                        current_line_length: current_line_len,
                    };
                    self.format_recursive::<T>(f, &new_params, &new_indices, depth + 1)?;
                }
            }
        }

        write!(f, "]")?;
        Ok(())
    }

    /// Helper function to format a value to string for length calculation
    fn format_value_to_string<T: std::fmt::Display>(&self, value: &T) -> String {
        match self.dtype() {
            DType::Fp16 | DType::Fp32 | DType::Fp64 | DType::Bf16 => {
                let value_str = format!("{value}");
                if let Ok(float_val) = value_str.parse::<f64>() {
                    self.format_float_to_string(float_val)
                } else {
                    value_str
                }
            }
            _ => format!("{value}"),
        }
    }

    /// Format a single value with proper float precision handling
    fn format_single_value<T: std::fmt::Display>(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        value: &T,
    ) -> std::fmt::Result {
        match self.dtype() {
            DType::Fp16 | DType::Fp32 | DType::Fp64 | DType::Bf16 => {
                self.format_float_value(f, value)
            }
            _ => write!(f, "{value}"),
        }
    }

    /// Format floating point values with precision control
    fn format_float_value<T: std::fmt::Display>(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        value: &T,
    ) -> std::fmt::Result {
        let value_str = format!("{value}");
        if let Ok(float_val) = value_str.parse::<f64>() {
            if f.width().is_some() || f.precision().is_some() {
                let precision = f.precision().unwrap_or(6);
                let width = f.width().unwrap_or(precision + 4);
                write!(f, "{float_val:>width$.precision$}")
            } else {
                self.format_float_smart(f, float_val)
            }
        } else {
            write!(f, "{value}")
        }
    }

    /// Smart float formatting: show minimal meaningful digits
    fn format_float_smart(&self, f: &mut std::fmt::Formatter<'_>, value: f64) -> std::fmt::Result {
        if value.fract().abs() < 1e-10 {
            write!(f, "{}.", value as i64)
        } else {
            let formatted = format!("{value:.6}");
            let trimmed = formatted.trim_end_matches('0').trim_end_matches('.');
            if !trimmed.contains('.') {
                write!(f, "{trimmed}.")
            } else {
                write!(f, "{trimmed}")
            }
        }
    }

    /// Format float to string for length calculation
    fn format_float_to_string(&self, value: f64) -> String {
        if value.fract().abs() < 1e-10 {
            format!("{}.", value as i64)
        } else {
            let formatted = format!("{value:.6}");
            let trimmed = formatted.trim_end_matches('0').trim_end_matches('.');
            if !trimmed.contains('.') {
                format!("{trimmed}.")
            } else {
                trimmed.to_string()
            }
        }
    }
}

// Implement formatting methods for TensorView
impl TensorView<'_> {
    /// Format tensor data for debug output with intelligent truncation
    fn format_data(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let _shape = self.shape().as_slice();
        let total_elements = self.numel();

        // Threshold for showing all elements vs truncating
        const SMALL_TENSOR_THRESHOLD: usize = 100;
        const MAX_ELEMENTS_PER_DIM: usize = 6;

        if total_elements == 0 {
            return write!(f, "[]");
        }

        if total_elements <= SMALL_TENSOR_THRESHOLD {
            // Show all elements for small tensors
            self.format_all_data(f)
        } else {
            // Show truncated view for large tensors
            self.format_truncated_data(f, MAX_ELEMENTS_PER_DIM)
        }
    }

    /// Get terminal width for better formatting (fallback to 80 if unavailable)
    fn get_terminal_width(&self) -> usize {
        // Simple fallback without external dependencies
        80
    }

    /// Format all tensor data (for small tensors)
    fn format_all_data(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.dtype() {
            DType::Bool => self.format_typed_data::<bool>(f, false, 0),
            DType::Int8 => self.format_typed_data::<i8>(f, false, 0),
            DType::Int16 => self.format_typed_data::<i16>(f, false, 0),
            DType::Int32 => self.format_typed_data::<i32>(f, false, 0),
            DType::Int64 => self.format_typed_data::<i64>(f, false, 0),
            DType::Uint8 => self.format_typed_data::<u8>(f, false, 0),
            DType::Uint16 => self.format_typed_data::<u16>(f, false, 0),
            DType::Uint32 => self.format_typed_data::<u32>(f, false, 0),
            DType::Uint64 => self.format_typed_data::<u64>(f, false, 0),
            DType::Fp16 => self.format_typed_data::<f16>(f, false, 0),
            DType::Fp32 => self.format_typed_data::<f32>(f, false, 0),
            DType::Fp64 => self.format_typed_data::<f64>(f, false, 0),
            DType::Bf16 => self.format_typed_data::<bf16>(f, false, 0),
            _ => Err(std::fmt::Error),
        }
    }

    /// Format truncated tensor data (for large tensors)
    fn format_truncated_data(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        max_per_dim: usize,
    ) -> std::fmt::Result {
        match self.dtype() {
            DType::Bool => self.format_typed_data::<bool>(f, true, max_per_dim),
            DType::Int8 => self.format_typed_data::<i8>(f, true, max_per_dim),
            DType::Int16 => self.format_typed_data::<i16>(f, true, max_per_dim),
            DType::Int32 => self.format_typed_data::<i32>(f, true, max_per_dim),
            DType::Int64 => self.format_typed_data::<i64>(f, true, max_per_dim),
            DType::Uint8 => self.format_typed_data::<u8>(f, true, max_per_dim),
            DType::Uint16 => self.format_typed_data::<u16>(f, true, max_per_dim),
            DType::Uint32 => self.format_typed_data::<u32>(f, true, max_per_dim),
            DType::Uint64 => self.format_typed_data::<u64>(f, true, max_per_dim),
            DType::Fp16 => self.format_typed_data::<f16>(f, true, max_per_dim),
            DType::Fp32 => self.format_typed_data::<f32>(f, true, max_per_dim),
            DType::Fp64 => self.format_typed_data::<f64>(f, true, max_per_dim),
            DType::Bf16 => self.format_typed_data::<bf16>(f, true, max_per_dim),
            _ => Err(std::fmt::Error),
        }
    }

    /// Format typed tensor data with optional truncation
    fn format_typed_data<T: TensorElement + std::fmt::Display>(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        truncate: bool,
        max_per_dim: usize,
    ) -> std::fmt::Result {
        let shape = self.shape().as_slice();

        if shape.is_empty() {
            // Scalar tensor
            let value = self.at::<T>(crate::Shape::empty());
            return self.format_single_value(f, &value);
        }

        let terminal_width = self.get_terminal_width();
        let params = FormatParams {
            shape,
            truncate,
            max_per_dim,
            line_width: terminal_width,
            current_line_length: 9, // Initial indentation for "  data: ["
        };

        self.format_recursive::<T>(f, &params, &[], 0)
    }

    /// Recursively format multi-dimensional tensor data with line width control
    fn format_recursive<T: TensorElement + std::fmt::Display>(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        params: &FormatParams,
        indices: &[usize],
        depth: usize,
    ) -> std::fmt::Result {
        let FormatParams {
            shape,
            truncate,
            max_per_dim,
            line_width,
            current_line_length,
        } = params;

        if shape.is_empty() {
            return Ok(());
        }

        let current_dim = shape[0];
        let remaining_shape = &shape[1..];
        let is_1d = remaining_shape.is_empty();
        let is_2d = remaining_shape.len() == 1;
        let is_outermost = depth == 0;
        let total_dims = shape.len() + depth;

        write!(f, "[")?;
        let mut current_line_len = *current_line_length + 1;

        let (show_count, need_ellipsis) = if *truncate && current_dim > *max_per_dim {
            (*max_per_dim / 2, true)
        } else {
            (current_dim, false)
        };

        // Show first elements
        for i in 0..show_count.min(current_dim) {
            if i > 0 {
                if is_1d {
                    // 1D tensor: check line width for smart wrapping
                    if current_line_len > line_width.saturating_sub(20) {
                        write!(f, ",\n         ")?; // 9 spaces to align with "  data: ["
                        current_line_len = 9;
                    } else {
                        write!(f, ", ")?;
                        current_line_len += 2;
                    }
                } else if is_outermost {
                    // Multi-dimensional outermost: newline with proper indentation
                    write!(f, ",\n         ")?; // 9 spaces to align with "  data: ["
                    current_line_len = 9;
                } else if total_dims >= 3 && is_2d {
                    // For 3D+ tensors, put each 2D slice on new line with extra indentation
                    let indent = " ".repeat(9 + depth * 2 - 1);
                    write!(f, ",\n{indent}")?;
                    current_line_len = 9 + depth * 2;
                } else {
                    // Inner dimensions: align to the position of the innermost '['
                    let indent = " ".repeat(9 + depth + 1);
                    write!(f, ",\n{indent}")?;
                    current_line_len = 9 + depth + 1;
                }
            }

            let mut new_indices = indices.to_vec();
            new_indices.push(i);

            if remaining_shape.is_empty() {
                // Leaf element - access data through TensorView's at method
                let indices_array = crate::Shape::from_slice(&new_indices);
                let value = self.at::<T>(indices_array);
                let value_str = self.format_value_to_string(&value);
                current_line_len += value_str.len();
                self.format_single_value(f, &value)?;
            } else {
                // Recursive call for sub-tensor
                let new_params = FormatParams {
                    shape: remaining_shape,
                    truncate: *truncate,
                    max_per_dim: *max_per_dim,
                    line_width: *line_width,
                    current_line_length: current_line_len,
                };
                self.format_recursive::<T>(f, &new_params, &new_indices, depth + 1)?;
            }
        }

        // Show ellipsis if truncated
        if need_ellipsis && current_dim > *max_per_dim {
            if show_count > 0 {
                if is_1d {
                    if current_line_len > line_width.saturating_sub(20) {
                        write!(f, ",\n         ")?;
                        current_line_len = 9;
                    } else {
                        write!(f, ", ")?;
                        current_line_len += 2;
                    }
                } else if is_outermost {
                    write!(f, ",\n         ")?;
                    current_line_len = 9;
                } else if total_dims >= 3 && is_2d {
                    let indent = " ".repeat(9 + depth * 2);
                    write!(f, ",\n{indent}")?;
                    current_line_len = 9 + depth * 2;
                } else {
                    let indent = " ".repeat(9 + depth + 1);
                    write!(f, ",\n{indent}")?;
                    current_line_len = 9 + depth + 1;
                }
            }
            write!(f, "...")?;
            current_line_len += 3;

            // Show last elements
            let last_start = current_dim - (*max_per_dim - show_count);
            for i in last_start..current_dim {
                if is_1d {
                    if current_line_len > line_width.saturating_sub(20) {
                        write!(f, ",\n         ")?;
                        current_line_len = 9;
                    } else {
                        write!(f, ", ")?;
                        current_line_len += 2;
                    }
                } else if is_outermost {
                    write!(f, ",\n         ")?;
                    current_line_len = 9;
                } else if total_dims >= 3 && is_2d {
                    let indent = " ".repeat(9 + depth * 2);
                    write!(f, ",\n{indent}")?;
                    current_line_len = 9 + depth * 2;
                } else {
                    let indent = " ".repeat(9 + depth + 1);
                    write!(f, ",\n{indent}")?;
                    current_line_len = 9 + depth + 1;
                }

                let mut new_indices = indices.to_vec();
                new_indices.push(i);

                if remaining_shape.is_empty() {
                    // Leaf element - access data through TensorView's at method
                    let indices_array = crate::Shape::from_slice(&new_indices);
                    let value = self.at::<T>(indices_array);
                    let value_str = self.format_value_to_string(&value);
                    current_line_len += value_str.len();
                    self.format_single_value(f, &value)?;
                } else {
                    // Recursive call for sub-tensor
                    let new_params = FormatParams {
                        shape: remaining_shape,
                        truncate: *truncate,
                        max_per_dim: *max_per_dim,
                        line_width: *line_width,
                        current_line_length: current_line_len,
                    };
                    self.format_recursive::<T>(f, &new_params, &new_indices, depth + 1)?;
                }
            }
        }

        write!(f, "]")?;
        Ok(())
    }

    /// Helper function to format a value to string for length calculation
    fn format_value_to_string<T: std::fmt::Display>(&self, value: &T) -> String {
        match self.dtype() {
            DType::Fp16 | DType::Fp32 | DType::Fp64 | DType::Bf16 => {
                let value_str = format!("{value}");
                if let Ok(float_val) = value_str.parse::<f64>() {
                    self.format_float_to_string(float_val)
                } else {
                    value_str
                }
            }
            _ => format!("{value}"),
        }
    }

    /// Format a single value with proper float precision handling
    fn format_single_value<T: std::fmt::Display>(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        value: &T,
    ) -> std::fmt::Result {
        match self.dtype() {
            DType::Fp16 | DType::Fp32 | DType::Fp64 | DType::Bf16 => {
                self.format_float_value(f, value)
            }
            _ => write!(f, "{value}"),
        }
    }

    /// Format floating point values with precision control
    fn format_float_value<T: std::fmt::Display>(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        value: &T,
    ) -> std::fmt::Result {
        let value_str = format!("{value}");
        if let Ok(float_val) = value_str.parse::<f64>() {
            if f.width().is_some() || f.precision().is_some() {
                let precision = f.precision().unwrap_or(6);
                let width = f.width().unwrap_or(precision + 4);
                write!(f, "{float_val:>width$.precision$}")
            } else {
                self.format_float_smart(f, float_val)
            }
        } else {
            write!(f, "{value}")
        }
    }

    /// Smart float formatting: show minimal meaningful digits
    fn format_float_smart(&self, f: &mut std::fmt::Formatter<'_>, value: f64) -> std::fmt::Result {
        if value.fract().abs() < 1e-10 {
            write!(f, "{}.", value as i64)
        } else {
            let formatted = format!("{value:.6}");
            let trimmed = formatted.trim_end_matches('0').trim_end_matches('.');
            if !trimmed.contains('.') {
                write!(f, "{trimmed}.")
            } else {
                write!(f, "{trimmed}")
            }
        }
    }

    /// Format float to string for length calculation
    fn format_float_to_string(&self, value: f64) -> String {
        if value.fract().abs() < 1e-10 {
            format!("{}.", value as i64)
        } else {
            let formatted = format!("{value:.6}");
            let trimmed = formatted.trim_end_matches('0').trim_end_matches('.');
            if !trimmed.contains('.') {
                format!("{trimmed}.")
            } else {
                trimmed.to_string()
            }
        }
    }
}
