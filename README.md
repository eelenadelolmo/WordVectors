# Annotation EN-ES mapper and expectation-based Thematic Progression annotation


### Installation

Clone the repo and execute the installation command:
`pip install -r requirements.txt`


### Usage

- English version: run `thematic_progression_annotator_en.py` and upload your files to http://0.0.0.0:5003/upload-tp-ann.
- Spanish version: run `thematic_progression_annotator.py` and upload your files to http://0.0.0.0:5003/upload-frame-ann-en.

    The Spanish pipeline begins with the SRL annotation mapper and the English version beging from the Thematic Progression annotator.

- **Annotation mapper**
    - Accepted input files: English annotated files with FrameNet SRL files (output of the previous module) and the original Spanish sentences files (output of the translation module in the es folder).
    - The output consists of ne JSON file for every sentence containing the Spanish SRL annotations.

- **Thematic Progression annotator**
    - Accepted input files: SRL annotated frames in JSON (output of the SRL module) and Grew annotated corresponding files for every text (output of the thematic annotator module). The names of the corresponding files for every sentence must match, but there is one file per every sentence of the text with its SRL annotation.
    - The output consists of:
        - xml folder with the raw data in XML.
        - png folder with the graphical output of the thematic progression of the texts.
        - html folder with the visual output of the thematic progression of the texts (the text, plus the concepst, plus its mentions along the themes and rhemes).

____


This project is a part of a PhD thesis carried out at the Department of Linguistics of the Complutense University of Madrid (https://www.ucm.es/linguistica/grado-linguistica) and is supported by the ILSA (Language-Driven Software and Applications) research group (http://ilsa.fdi.ucm.es/).
