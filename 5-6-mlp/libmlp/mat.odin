package mlp

import "core:fmt"
// Row major matrix
Matrix :: struct {
	data: []f64,
	rows: int,
	cols: int,
}

stride :: #force_inline proc "contextless" (m: Matrix) -> int {
	return m.cols
}

mat_full_copy :: proc(m: Matrix) -> Matrix {
	data := make([]f64, len(m.data))
	copy(data, m.data)
	matr := Matrix {
		data = data,
		rows = m.rows,
		cols = m.cols,
	}

	return matr
}

matmul_dst :: proc(
	lhs, rhs: Matrix,
	bias: []f64 = nil,
	dst: Matrix,
	scale: f64 = 1.0,
	transpose_lhs: bool = false,
	transpose_rhs: bool = false,
) {

	lhs_rows := transpose_lhs ? lhs.cols : lhs.rows
	lhs_cols := transpose_lhs ? lhs.rows : lhs.cols

	rhs_rows := transpose_rhs ? rhs.cols : rhs.rows
	rhs_cols := transpose_rhs ? rhs.rows : rhs.cols


	if lhs_cols != rhs_rows {
		fmt.printfln(
			"matrix shapes diverge: lhs %d*%d, rhs %d*%d",
			lhs.rows,
			lhs.cols,
			rhs.rows,
			rhs.cols,
		)
		return
	}

	for i in 0 ..< lhs_rows {
		for j in 0 ..< rhs_cols {
			agg := len(bias) > 0 ? bias[j] : 0.0
			for k in 0 ..< lhs_cols {
				lhs_el_idx := transpose_lhs ? k * stride(lhs) + i : i * stride(lhs) + k
				first_el := lhs.data[lhs_el_idx]

				rhs_el_idx := transpose_rhs ? j * stride(rhs) + k : k * stride(rhs) + j
				sec_el := rhs.data[rhs_el_idx]
				agg += first_el * sec_el
			}

			dst.data[i * stride(dst) + j] = agg * scale
		}
	}
}


matmul_ret :: proc(
	lhs, rhs: Matrix,
	bias: []f64 = nil,
	scale: f64 = 1.0,
	transpose_lhs: bool = false,
	transpose_rhs: bool = false,
) -> Matrix {
	data_size := lhs.rows * rhs.cols
	data := make([]f64, data_size)

	m := Matrix{data, lhs.rows, rhs.cols}

	matmul_dst(lhs, rhs, bias, m, scale, transpose_lhs, transpose_rhs)

	return m
}

matmul :: proc {
	matmul_ret,
	matmul_dst,
}
