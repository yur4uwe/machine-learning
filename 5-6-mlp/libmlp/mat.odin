package mlp

import "base:intrinsics"
import "core:fmt"
import "core:simd"

simd_lane :: #simd[8]f32

load_into_simd_reg :: #force_inline proc "contextless" (s: []f32, idx: int) -> simd_lane {
	return intrinsics.unaligned_load((^simd_lane)(&s[idx]))
}

load_from_simd_reg :: #force_inline proc "contextless" (s: []f32, idx: int, v: simd_lane) {
	intrinsics.unaligned_store((^simd_lane)(&s[idx]), v)
}

// Row major matrix
Matrix :: struct {
	data: []f32,
	rows: int,
	cols: int,
}

Transposed_Matrix :: distinct Matrix

transpose :: #force_inline proc "contextless" (m: Matrix) -> Transposed_Matrix {
	return Transposed_Matrix(m)
}

stride :: #force_inline proc "contextless" (m: Matrix) -> int {
	return m.cols
}

mat_full_copy :: proc(m: Matrix) -> Matrix {
	data := make([]f32, len(m.data))
	copy(data, m.data)
	matr := Matrix {
		data = data,
		rows = m.rows,
		cols = m.cols,
	}

	return matr
}

matmul_nn :: proc(
	lhs, rhs: Matrix,
	bias: []f32 = nil,
	dst: Matrix,
	scale: f32 = 1.0,
) {
	M := lhs.rows
	K := lhs.cols
	N := rhs.cols

	if K != rhs.rows {
		fmt.printfln(
			"matrix shapes diverge: lhs %d*%d, rhs %d*%d",
			lhs.rows,
			lhs.cols,
			rhs.rows,
			rhs.cols,
		)
		return
	}

	has_bias := len(bias) > 0
	for i in 0 ..< M {
		dst_row := dst.data[i * N : i * N + N]
		if has_bias {
			copy(dst_row, bias)
		} else {
			for &v in dst_row {
				v = 0.0
			}
		}

		lhs_row := lhs.data[i * K : i * K + K]
		for k in 0 ..< K {
			a := lhs_row[k]
			a_vec := simd_lane(a)
			rhs_row := rhs.data[k * N : k * N + N]
			j := 0
			for ; j + 8 <= N; j += 8 {
				d := load_into_simd_reg(dst_row, j)
				r := load_into_simd_reg(rhs_row, j)
				load_from_simd_reg(dst_row, j, simd.fused_mul_add(a_vec, r, d))
			}
			for ; j < N; j += 1 {
				dst_row[j] += a * rhs_row[j]
			}
		}

		if scale != 1.0 {
			s_vec := simd_lane(scale)
			j := 0
			for ; j + 8 <= N; j += 8 {
				load_from_simd_reg(dst_row, j, load_into_simd_reg(dst_row, j) * s_vec)
			}
			for ; j < N; j += 1 {
				dst_row[j] *= scale
			}
		}
	}
}

matmul_tn :: proc(
	lhs: Transposed_Matrix,
	rhs: Matrix,
	bias: []f32 = nil,
	dst: Matrix,
	scale: f32 = 1.0,
) {
	M := lhs.cols
	K := lhs.rows
	N := rhs.cols

	if K != rhs.rows {
		fmt.printfln(
			"matrix shapes diverge: lhs^T %d*%d, rhs %d*%d",
			M,
			K,
			rhs.rows,
			rhs.cols,
		)
		return
	}

	if len(bias) > 0 {
		for i in 0 ..< M {
			copy(dst.data[i * N : i * N + N], bias)
		}
	} else {
		for &v in dst.data {
			v = 0.0
		}
	}

	for k in 0 ..< K {
		lhs_row := lhs.data[k * M : k * M + M]
		rhs_row := rhs.data[k * N : k * N + N]
		for i in 0 ..< M {
			a := lhs_row[i]
			a_vec := simd_lane(a)
			dst_row := dst.data[i * N : i * N + N]
			j := 0
			for ; j + 8 <= N; j += 8 {
				d := load_into_simd_reg(dst_row, j)
				r := load_into_simd_reg(rhs_row, j)
				load_from_simd_reg(dst_row, j, simd.fused_mul_add(a_vec, r, d))
			}
			for ; j < N; j += 1 {
				dst_row[j] += a * rhs_row[j]
			}
		}
	}

	if scale != 1.0 {
		s_vec := simd_lane(scale)
		n_total := len(dst.data)
		idx := 0
		for ; idx + 8 <= n_total; idx += 8 {
			load_from_simd_reg(dst.data, idx, load_into_simd_reg(dst.data, idx) * s_vec)
		}
		for ; idx < n_total; idx += 1 {
			dst.data[idx] *= scale
		}
	}
}

matmul_nt :: proc(
	lhs: Matrix,
	rhs: Transposed_Matrix,
	bias: []f32 = nil,
	dst: Matrix,
	scale: f32 = 1.0,
) {
	M := lhs.rows
	K := lhs.cols
	N := rhs.rows

	if K != rhs.cols {
		fmt.printfln(
			"matrix shapes diverge: lhs %d*%d, rhs^T %d*%d",
			lhs.rows,
			lhs.cols,
			rhs.cols,
			N,
		)
		return
	}

	has_bias := len(bias) > 0
	for i in 0 ..< M {
		lhs_row := lhs.data[i * K : i * K + K]
		dst_row := dst.data[i * N : i * N + N]
		for j in 0 ..< N {
			rhs_row := rhs.data[j * K : j * K + K]
			acc := simd_lane(0.0)
			k := 0
			for ; k + 8 <= K; k += 8 {
				l := load_into_simd_reg(lhs_row, k)
				r := load_into_simd_reg(rhs_row, k)
				acc = simd.fused_mul_add(l, r, acc)
			}
			agg := (has_bias ? bias[j] : 0.0) + simd.reduce_add_ordered(acc)
			for ; k < K; k += 1 {
				agg += lhs_row[k] * rhs_row[k]
			}
			dst_row[j] = agg * scale
		}
	}
}

matmul_ret :: proc(
	lhs, rhs: Matrix,
	bias: []f32 = nil,
	scale: f32 = 1.0,
	transpose_lhs: bool = false,
	transpose_rhs: bool = false,
) -> Matrix {
	rows := transpose_lhs ? lhs.cols : lhs.rows
	cols := transpose_rhs ? rhs.rows : rhs.cols
	data := make([]f32, rows * cols)
	m := Matrix{data, rows, cols}

	switch {
	case !transpose_lhs && !transpose_rhs:
		matmul_nn(lhs, rhs, bias, m, scale)
	case transpose_lhs && !transpose_rhs:
		matmul_tn(transpose(lhs), rhs, bias, m, scale)
	case !transpose_lhs && transpose_rhs:
		matmul_nt(lhs, transpose(rhs), bias, m, scale)
	case:
		fmt.println("matmul_ret: both operands transposed is not supported")
	}

	return m
}

matmul :: proc {
	matmul_ret,
	matmul_nn,
	matmul_tn,
	matmul_nt,
}
