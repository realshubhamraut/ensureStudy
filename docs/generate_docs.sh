#!/bin/bash
# Generate PDF documentation with rendered Mermaid diagrams
# Requirements: pandoc, mermaid-filter, LaTeX

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOC_DIR="$SCRIPT_DIR"
OUTPUT_DIR="$SCRIPT_DIR/output"
DIAGRAMS_DIR="$OUTPUT_DIR/diagrams"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$DIAGRAMS_DIR"

echo "======================================"
echo "EnsureStudy Documentation Generator"
echo "With Mermaid Diagram Rendering"
echo "======================================"

# Input file
INPUT_FILE="$DOC_DIR/EnsureStudy_Technical_Documentation.md"
PROCESSED_FILE="$OUTPUT_DIR/processed_doc.md"
OUTPUT_PDF="$OUTPUT_DIR/EnsureStudy_Technical_Documentation.pdf"
OUTPUT_DOCX="$OUTPUT_DIR/EnsureStudy_Technical_Documentation.docx"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi

# Check for Pandoc
if ! command -v pandoc &> /dev/null; then
    echo "Error: Pandoc is not installed. Install with: brew install pandoc"
    exit 1
fi

# Check for Node.js (needed for mermaid-cli)
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed"
    exit 1
fi

# Check for mmdc
MMDC_PATH=""
if command -v mmdc &> /dev/null; then
    MMDC_PATH="mmdc"
elif [ -f "/opt/homebrew/lib/node_modules/mermaid-filter/node_modules/.bin/mmdc" ]; then
    MMDC_PATH="/opt/homebrew/lib/node_modules/mermaid-filter/node_modules/.bin/mmdc"
elif [ -f "/opt/homebrew/lib/node_modules/@mermaid-js/mermaid-cli/node_modules/.bin/mmdc" ]; then
    MMDC_PATH="/opt/homebrew/lib/node_modules/@mermaid-js/mermaid-cli/node_modules/.bin/mmdc"
else
    echo "Installing @mermaid-js/mermaid-cli..."
    npm install -g @mermaid-js/mermaid-cli
    MMDC_PATH="mmdc"
fi

echo ""
echo "Step 1: Extracting and rendering Mermaid diagrams..."
echo "-----------------------------------------------------"

# Create a temporary file to track diagram counter
COUNTER=0
cp "$INPUT_FILE" "$PROCESSED_FILE"

# Remove emoji that cause LaTeX issues
sed -i '' 's/👍/[thumbs up]/g; s/👎/[thumbs down]/g' "$PROCESSED_FILE" 2>/dev/null || true

# Extract and render each mermaid diagram
while IFS= read -r -d '' MERMAID_BLOCK; do
    COUNTER=$((COUNTER + 1))
    DIAGRAM_FILE="$DIAGRAMS_DIR/diagram_${COUNTER}"
    
    echo "  Rendering diagram $COUNTER..."
    
    # Write mermaid code to temp file
    echo "$MERMAID_BLOCK" > "${DIAGRAM_FILE}.mmd"
    
    # Render to PNG using mmdc
    if [ -n "$MMDC_PATH" ]; then
        $MMDC_PATH -i "${DIAGRAM_FILE}.mmd" -o "${DIAGRAM_FILE}.png" -b white -w 800 2>/dev/null || {
            echo "    Warning: Failed to render diagram $COUNTER, using placeholder"
            echo "[Diagram $COUNTER - See HTML version for interactive diagram]" > "${DIAGRAM_FILE}.txt"
            continue
        }
    fi
done < <(grep -Pzo '(?s)```mermaid\n\K.*?(?=\n```)' "$INPUT_FILE" 2>/dev/null | tr '\0' '\n' || true)

# Try alternative extraction method if grep -P doesn't work
if [ $COUNTER -eq 0 ]; then
    echo "  Using alternative diagram extraction..."
    
    # Use awk to extract mermaid blocks
    awk '/```mermaid/{flag=1; next} /```/{if(flag) {print "---BLOCK_END---"; flag=0}} flag' "$INPUT_FILE" | \
    awk -v RS='---BLOCK_END---' -v ORS='' -v dir="$DIAGRAMS_DIR" -v mmdc="$MMDC_PATH" '
    NF {
        counter++
        filename = dir "/diagram_" counter ".mmd"
        print $0 > filename
        close(filename)
        
        # Render with mmdc
        if (mmdc != "") {
            cmd = mmdc " -i " filename " -o " dir "/diagram_" counter ".png -b white -w 800 2>/dev/null"
            system(cmd)
        }
        print "  Rendered diagram " counter
    }'
    
    COUNTER=$(ls -1 "$DIAGRAMS_DIR"/*.mmd 2>/dev/null | wc -l | tr -d ' ')
fi

echo "  Rendered $COUNTER diagrams"

echo ""
echo "Step 2: Replacing Mermaid blocks with images..."
echo "------------------------------------------------"

# Create the processed file with diagram images
python3 << 'PYTHON_SCRIPT'
import re
import os

input_file = os.environ.get('INPUT_FILE', 'EnsureStudy_Technical_Documentation.md')
output_file = os.environ.get('PROCESSED_FILE', 'output/processed_doc.md')
diagrams_dir = os.environ.get('DIAGRAMS_DIR', 'output/diagrams')

with open(input_file, 'r') as f:
    content = f.read()

# Remove emoji that cause LaTeX issues
content = content.replace('👍', '[thumbs-up]')
content = content.replace('👎', '[thumbs-down]')

# Find all mermaid blocks and replace with images
counter = 0
def replace_mermaid(match):
    global counter
    counter += 1
    png_file = os.path.join(diagrams_dir, f'diagram_{counter}.png')
    if os.path.exists(png_file):
        # Use absolute path for pandoc
        abs_path = os.path.abspath(png_file)
        return f'\n![Diagram {counter}]({abs_path})\n'
    else:
        return f'\n*[Diagram {counter} - rendering failed]*\n'

# Match ```mermaid ... ``` blocks
pattern = r'```mermaid\n.*?```'
content = re.sub(pattern, replace_mermaid, content, flags=re.DOTALL)

with open(output_file, 'w') as f:
    f.write(content)

print(f"  Processed {counter} diagram references")
PYTHON_SCRIPT

export INPUT_FILE PROCESSED_FILE DIAGRAMS_DIR
python3 -c "
import re
import os

input_file = '$INPUT_FILE'
output_file = '$PROCESSED_FILE'
diagrams_dir = '$DIAGRAMS_DIR'

with open(input_file, 'r') as f:
    content = f.read()

# Remove emoji
content = content.replace('👍', '[thumbs-up]')
content = content.replace('👎', '[thumbs-down]')

# Replace mermaid blocks with images
counter = [0]
def replace_mermaid(match):
    counter[0] += 1
    png_file = os.path.join(diagrams_dir, f'diagram_{counter[0]}.png')
    if os.path.exists(png_file):
        abs_path = os.path.abspath(png_file)
        return f'\n![Diagram {counter[0]}]({abs_path})\n'
    else:
        return f'\n*[Diagram {counter[0]}]*\n'

pattern = r'\`\`\`mermaid\n.*?\`\`\`'
content = re.sub(pattern, replace_mermaid, content, flags=re.DOTALL)

with open(output_file, 'w') as f:
    f.write(content)

print(f'  Processed {counter[0]} diagram references')
"

echo ""
echo "Step 3: Generating PDF with LaTeX..."
echo "------------------------------------"

# Check for LaTeX
if command -v xelatex &> /dev/null; then
    PDF_ENGINE="xelatex"
elif command -v pdflatex &> /dev/null; then
    PDF_ENGINE="pdflatex"
else
    echo "Error: LaTeX not installed. Install with: brew install --cask mactex-no-gui"
    exit 1
fi

cd "$OUTPUT_DIR"

pandoc "$PROCESSED_FILE" \
    -o "$OUTPUT_PDF" \
    --from markdown \
    --pdf-engine="$PDF_ENGINE" \
    --toc \
    --toc-depth=3 \
    --number-sections \
    -V geometry:margin=1in \
    -V fontsize=12pt \
    -V mainfont="Times New Roman" \
    -V colorlinks=true \
    -V linkcolor=blue \
    -V urlcolor=blue \
    -V documentclass=report \
    --resource-path="$DIAGRAMS_DIR" \
    2>&1 | grep -v "^\\[WARNING\\]" | head -20

if [ -f "$OUTPUT_PDF" ]; then
    echo "✅ Generated: $OUTPUT_PDF"
    echo "   Size: $(du -h "$OUTPUT_PDF" | cut -f1)"
else
    echo "❌ PDF generation failed"
fi

echo ""
echo "Step 4: Generating DOCX..."
echo "--------------------------"

pandoc "$PROCESSED_FILE" \
    -o "$OUTPUT_DOCX" \
    --from markdown \
    --to docx \
    --toc \
    --toc-depth=3 \
    --number-sections \
    --resource-path="$DIAGRAMS_DIR" \
    2>&1 | grep -v "^\\[WARNING\\]"

if [ -f "$OUTPUT_DOCX" ]; then
    echo "✅ Generated: $OUTPUT_DOCX"
else
    echo "❌ DOCX generation failed"
fi

echo ""
echo "======================================"
echo "Documentation generation complete!"
echo "======================================"
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"/*.pdf "$OUTPUT_DIR"/*.docx 2>/dev/null
echo ""
echo "Diagrams rendered: $(ls -1 "$DIAGRAMS_DIR"/*.png 2>/dev/null | wc -l | tr -d ' ')"
