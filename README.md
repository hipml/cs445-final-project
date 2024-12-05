# Project Name
**CS445 Fall 2024 Final Project** University of Illinois  

<ins>Group members:</ins>  
* Matt Poteshman (**mrp12**)  
* Paul Lambert (**lamber10**)  

## Background
Project information and ideas. Texture transfer, but from talented artists! 

## Instructions
To build and run

```bash
python3 -m venv myenv  
source myenv/bin/activate  
pip install -r requirements.txt
```
### Neural Style Transfer
```python src/nst.py```

### CycleGAN
To train: ```python project.py --train --style_dir "images/art/vangogh/" --epochs 100 --batch_size 1 --lr 0.0002```  
Inference: ```python project.py --input_image "images/input/input.jpg" --output_image "images/output/styled_vangogh.jpg"```
### Other implementation


## Sources

### Art Sources

[Van Gogh's works](https://www.nga.gov/collection/artist-info.1349.html#works)

#### Currently not used
* https://drive.google.com/drive/folders/1CglMyDFXJFNpDt3ebstOPYVwnVvMNv6g
* https://www.reddit.com/r/DataHoarder/comments/d0wuae/50k_images_from_the_art_institute_of_chicago/
* https://www.reddit.com/r/TheFrame/comments/10cu8hg/over_400_4k_opensource_artworks_from_around_the/
