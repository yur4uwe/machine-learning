package mlp

import "core:os"
import "base:runtime"

@(export)
model_save_params :: proc "c" (handle: rawptr, path: cstring) -> bool {
	context = runtime.default_context()
	m := (^Model)(handle)
	f, err := os.open(string(path), flags={.Write, .Create})
	if err != nil do return false
	defer os.close(f)
	save_model(m^, f)
	return true
}

@(export)
model_load_params :: proc "c" (path: cstring) -> rawptr {
	context = runtime.default_context()
	f, file_err := os.open(string(path))
	if file_err != nil do return nil
	defer os.close(f)
	model, err := load_model(f)
	if err != nil do return nil

	handle := new(Model)
	handle^ = model
	return handle
}
