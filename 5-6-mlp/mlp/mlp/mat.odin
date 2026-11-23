package mlp

import "core:fmt"
// Row major matrix
Matrix :: struct {
	data:  []f64,
	shape: struct {
		row_cnt: int,
		col_cnt: int,
	},
}

major_access :: proc(m: ^Matrix, row_idx: int) -> ([]f64, bool) {
	if m.shape.row_cnt < row_idx {
		return nil, false
	}
	row_start := m.shape.col_cnt * row_idx
	row_end := row_start + m.shape.col_cnt
	return m.data[row_start:row_end], true
}

minor_access :: proc()

el_access :: proc(m: ^Matrix, row_idx: int, col_idx: int) -> (f64, bool) {
	if m.shape.row_cnt <= row_idx || m.shape.col_cnt <= col_idx {
		return 0.0, false
	}

	act_idx := m.shape.col_cnt * row_idx + col_idx

	fmt.printfln("access at: [%d][%d] with flattened %d", row_idx, col_idx, act_idx)

	return m.data[act_idx], true
}

el_update :: proc(m: ^Matrix, row_idx: int, col_idx: int, v: f64) -> bool {
	if m.shape.row_cnt <= row_idx || m.shape.col_cnt <= col_idx {
		return false
	}

	act_idx := m.shape.col_cnt * row_idx + col_idx

	fmt.printfln("update at: [%d][%d] with flattened %d", row_idx, col_idx, act_idx)

	m.data[act_idx] = v
	return true
}

stride :: #force_inline proc "contextless" (m: Matrix) -> int {
	return m.shape.col_cnt
}

matmul_dst :: proc(
	lhs, rhs: Matrix,
	bias: []f64 = nil,
	dst: Matrix,
	scale: f64 = 1.0,
	transpose_lhs: bool = false,
	transpose_rhs: bool = false,
) {

	lhs_rows := transpose_lhs ? lhs.shape.col_cnt : lhs.shape.row_cnt
	lhs_cols := transpose_lhs ? lhs.shape.row_cnt : lhs.shape.col_cnt

	rhs_rows := transpose_rhs ? rhs.shape.col_cnt : rhs.shape.row_cnt
	rhs_cols := transpose_rhs ? rhs.shape.row_cnt : rhs.shape.col_cnt


	if lhs_cols != rhs_rows {
		fmt.printfln(
			"matrix shapes diverge: lhs %d*%d, rhs %d*%d",
			lhs.shape.row_cnt,
			lhs.shape.col_cnt,
			rhs.shape.row_cnt,
			rhs.shape.col_cnt,
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
	data_size := lhs.shape.row_cnt * rhs.shape.col_cnt
	data := make([]f64, data_size)

	m := Matrix{data, {lhs.shape.row_cnt, rhs.shape.col_cnt}}

	matmul_dst(lhs, rhs, bias, m, scale, transpose_lhs, transpose_rhs)

	return m
}

matmul :: proc {
	matmul_ret,
	matmul_dst,
}
