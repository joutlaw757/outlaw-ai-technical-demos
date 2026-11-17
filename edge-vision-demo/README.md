# Edge Vision Demo (new illustrative sample)

A lightweight, CPU-only image classification inference demonstration using a pretrained MobileNetV3 model. **This is a newly created demonstration, not evidence of prior production deployment or personal authorship of an earlier system.**

## Setup
```bash
python -m venv .venv
# activate environment
pip install -r requirements.txt
python classify.py path/to/image.jpg
```
First execution downloads public pretrained weights. No camera or proprietary datasets required. Output is the top five ImageNet predictions. CPU inference is not proof of an optimized embedded deployment.
