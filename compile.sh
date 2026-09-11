#!/usr/bin/env bash

usage() {
    echo "Usage: $0 <ipynb|html> <input_file> <output_file>"
    echo ""
    echo "Arguments:"
    echo "  1. type        Target format: 'ipynb' or 'html'"
    echo "  2. input_file  Path to source file (.py or .ipynb)"
    echo "  3. output_file Path to destination file (.ipynb or .html)"
    echo ""
    echo "Examples:"
    echo "  $0 ipynb 1-linear-regression/main.py 1-linear-regression/main.ipynb"
    echo "  $0 html 1-linear-regression/main.ipynb 1-linear-regression/main.html"
    echo "  $0 html 1-linear-regression/main.py 1-linear-regression/main.html"
    exit 1
}

if [ "$#" -ne 3 ]; then
    usage
fi

TARGET_TYPE="$1"
INPUT_PATH="$2"
OUTPUT_PATH="$3"

if [ ! -f "$INPUT_PATH" ]; then
    echo "Error: Input file '$INPUT_PATH' not found."
    exit 1
fi

OUTPUT_DIR="$(dirname "$OUTPUT_PATH")"
OUTPUT_FILE="$(basename "$OUTPUT_PATH")"
mkdir -p "$OUTPUT_DIR"

case "$TARGET_TYPE" in
    ipynb)
        echo "Converting '$INPUT_PATH' to notebook '$OUTPUT_PATH'..."
        uv run jupytext --to notebook --set-kernel machine-learning "$INPUT_PATH" -o "$OUTPUT_PATH"

        # Execute notebook in place to save outputs directly inside the .ipynb file
        EXEC_CWD="$(cd "$(dirname "$OUTPUT_PATH")" && pwd)"
        NOTEBOOK_NAME="$(basename "$OUTPUT_PATH" .ipynb)"

        echo "Executing notebook in '$EXEC_CWD' and updating '$NOTEBOOK_NAME' with outputs..."
        (
            cd "$EXEC_CWD"
            uv run --with nbconvert --with ipykernel jupyter nbconvert \
                --to notebook \
                --execute \
                --inplace \
                "$NOTEBOOK_NAME"
        )

        echo "Successfully created '$OUTPUT_PATH'."
        ;;

    html)
        SOURCE_NOTEBOOK="$INPUT_PATH"
        INPUT_DIR="$(cd "$(dirname "$INPUT_PATH")" && pwd)"

        # If input is a Python script, sync it to the corresponding .ipynb file first
        if [[ "$INPUT_PATH" == *.py ]]; then
            TARGET_IPYNB="${INPUT_DIR}/$(basename "$INPUT_PATH" .py).ipynb"
            echo "Syncing '$INPUT_PATH' to notebook '$TARGET_IPYNB'..."
            uv run jupytext --to notebook --set-kernel machine-learning "$INPUT_PATH" -o "$TARGET_IPYNB"
            SOURCE_NOTEBOOK="$TARGET_IPYNB"
        fi

        ABS_OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
        EXEC_CWD="$(cd "$(dirname "$SOURCE_NOTEBOOK")" && pwd)"
        NOTEBOOK_NAME="$(basename "$SOURCE_NOTEBOOK")"

        echo "Executing notebook in '$EXEC_CWD' and updating '$NOTEBOOK_NAME' with outputs..."
        (
            cd "$EXEC_CWD"
            # 1. Execute notebook in place to save outputs directly inside the .ipynb file
            uv run --with nbconvert --with ipykernel jupyter nbconvert \
                --to notebook \
                --execute \
                --inplace \
                "$NOTEBOOK_NAME"

            # 2. Convert executed notebook to HTML
            echo "Exporting executed notebook to HTML '$OUTPUT_PATH'..."
            uv run --with nbconvert --with ipykernel jupyter nbconvert \
                --to html \
                "$NOTEBOOK_NAME" \
                --output "$OUTPUT_FILE" \
                --output-dir "$ABS_OUTPUT_DIR"
        )

        echo "Successfully updated notebook and created HTML '$OUTPUT_PATH'."
        ;;
    *)
        echo "Error: Unsupported target type '$TARGET_TYPE'. Allowed types: ipynb, html."
        usage
        ;;
esac
