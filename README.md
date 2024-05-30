# Description
Code for my Master thesis in Political Science with focus on Computational Social Science at the University of Bamberg. It fine-tunes a XLM-R model using the PEFT-technique [AdaLoRa](https://arxiv.org/abs/2303.10512) on party manifesto data. The training data can be obtained from the Manifesto Project. After training, the model can be used to predict Comparative Agenda Project classes in parliamentary speech.

## Usage
It is recommended to build your own Docker-Image to use the model. Clone the repo and build the image using the following command:

```
cd thesis-code
docker build -t dlcontainer . 
```

Then run the container using:

```
docker run -it --rm --name adalora -v "$(pwd)":/app --gpus all dlcontainer
```

Do not forget to configure Docker to use your Nvidia GPU.
