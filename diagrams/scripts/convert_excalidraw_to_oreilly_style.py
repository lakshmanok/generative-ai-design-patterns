#!/usr/bin/env python
import base64
import xml.etree.ElementTree as ET

# This script assumes that you have this font file in
# the current directory
FONT_FILE="./GuardianSansCond-RegularIt.otf"
INPUT_DIR="../original"
OUTPUT_DIR=".."

"""
Converts an Excalidraw svg file to use O'Reilly font
Assumes that you have GuardianSansCond-RegularIt.otf
in the current directory.
"""
def convert_svg(infile: str, outfile: str):
    try:
        tree = ET.parse(infile)
        root = tree.getroot()

        # SVG namespace
        ns = {'svg': 'http://www.w3.org/2000/svg'}

        # Find all text elements and update their font-family
        for text_element in root.findall('.//svg:text', ns):
            text_element.set('font-family', "Guardian")

        # Add a font-face for the Guardian font
        with open(FONT_FILE, "rb") as ifp:
            font_data = base64.b64encode(ifp.read()).decode("utf-8")
            font_format = "opentype" # because it is otf

        # find the style tag tha defines the Excalidraw font
        # and replace that definition with the Guardian font
        for style_element in root.findall(".//svg:style", ns):
            if "@font-face" in style_element.text and "Excalifont" in style_element.text:
                print(f"{infile}: Replacing {style_element.text[:50].strip()} ...")
                style_element.text = f"""
                        @font-face {{
                            font-family: 'Guardian';
                            src: url(data:font/{font_format};base64,{font_data}) format('{font_format}');
                        }}
                """

        # Save the modified SVG
        tree.write(outfile, encoding="UTF-8", xml_declaration=True)
        print(f"Wrote out {outfile}")

    except Exception as e:
        print(f"Error handling {infile}: {e}")


"""
Converts all the svg files in the current directory to O'Reilly fonts
"""
if __name__ == '__main__':
    import glob
    import os
    # make sure that FONT_FILE exists
    if not os.path.exists(FONT_FILE):
        print(f"{FONT_FILE} missing. Please ask your O'Reilly editor to send you the font")

    # get all files that match *.svg
    input_files = glob.glob(os.path.join(INPUT_DIR, "*.svg"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for infile in input_files:
        print(infile)
        convert_svg(infile, os.path.join(OUTPUT_DIR, os.path.basename(infile)))