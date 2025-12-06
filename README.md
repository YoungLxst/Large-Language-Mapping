# Large-Language-Mapping

A real time séquentiel model for language classification

## Dataset

For this framework the dataset used is the uging face [common_laguage dataset](https://huggingface.co/datasets/speechbrain/common_language). It contain voice records of 45 diffenrents language.

**init**

execute:  
```cmd
python ./utils/load_data.py
````

to add the data file with all the data preprocesing that we need

## Warning

On running audio files. Torchcodec need ffmpeg DLL. Verify if your computer have them