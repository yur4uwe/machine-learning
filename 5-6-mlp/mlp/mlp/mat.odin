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

stride :: #force_inline proc(m: Matrix) -> int {
	return m.shape.col_cnt
}

matmul :: proc(lhs: Matrix, rhs: Matrix, bias: []f64 = nil) -> (Matrix, bool) {
	// 1 0
	// 0 1 * 1 0 0 0
	// 0 0   0 1 0 1
	// 1 1
	if lhs.shape.col_cnt != rhs.shape.row_cnt {
		fmt.printfln(
			"matrix shapes diverge: lhs %d*%d, rhs %d*%d",
			lhs.shape.row_cnt,
			lhs.shape.col_cnt,
			rhs.shape.row_cnt,
			rhs.shape.col_cnt,
		)
		return Matrix{}, false
	}

	data_size := lhs.shape.row_cnt * rhs.shape.col_cnt

	data, err := make([]f64, data_size)
	if err != nil {
		return Matrix{}, false
	}

	m := Matrix{data, {lhs.shape.row_cnt, rhs.shape.col_cnt}}

	for i in 0 ..< lhs.shape.row_cnt {
		for j in 0 ..< rhs.shape.col_cnt {
			agg := 0.0
			for k in 0 ..< lhs.shape.col_cnt {
				first_el := lhs.data[i * stride(lhs) + k]
				sec_el := rhs.data[k * stride(rhs) + j]
				agg += first_el * sec_el
			}

			m.data[i * stride(m) + j] = agg
		}
	}

	return m, true
}
