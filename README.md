# Description
Code for my Master thesis in Political Science with focus on Computational Social Science at the University of Bamberg. It fine-tunes a XLM-R model using the PEFT-technique AdaLoRa on party manifesto data. The training data can be obtained from the Manifesto Project. After training, the model can be used to predict Comparative Agenda Project classes in parliamentary speech.

## Usage
It is recommended to build your own Docker-Image to use the model. Clone the repo and build the image using the following command:
''
docker build -t dlContainer . 
''
Then run the container using:
``
docker run dlContainer
``
Do not forget to configure Docker to use your Nvidia GPU.
