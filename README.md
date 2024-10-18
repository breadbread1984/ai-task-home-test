# Answers

- The discriminator distinguishes between real samples and generated samples. src.models.modules.discriminators.Discriminator classifies images from MNIST as positive samples and classifies images generated by src.models.modules.generators.Generator as negative samples.

- Noise and labels are the noise input and condition input of the generator, respectively. Feeding sample from a standard gaussian distribution to noise and 5 to labels to generate an image of the number 5.

- First, load trained model to nn.Module without pytorch lightning dependency. Second, convert the model to onnx. Then, the onnx can be compiled into TensorRT engine and run on Nvidia inference card. Last but not least, the onnx can be also compiled and calibrated into quantized TensorRT engine.

- Training with multiple GPU can be achieved by data parallelism or model parallelism. Pytorch lightning can train model with data parallelism by setting the device number and the data parallelism strategy at Trainer's construction arguments.

# Training Results

## WanDB charts

### GAN on MNIST

- [wandb workspace](https://wandb.ai/yxie-lhi-hkqai/Tests/runs/se7mrkdr?nw=nwuseryxielhi)

- <img title="image generated by GAN" alt="image generated by GAN" src="/pics/mnist.png">

### DDPM on CIFAR10

- [wandb workspace](https://wandb.ai/yxie-lhi-hkqai/Tests/runs/m9uwf6b1?nw=nwuseryxielhi)

- <img title="image generated by DDPM" alt="image generated by DDPM" src="/pics/cifar10.png">

## Difficulties and Solutions

- The first difficulty I met is the figuring out how to use pytorch lightning, hydra and wandb. The problem is solved by reading document and search answers from ChatGPT.

- The second difficulty I met is unbalance convergence speeds between generator and discriminator. I solve the problem by add sigmoid at the end of the discriminator and use binary crossentropy to supervise discriminator to slow convergence of discriminator. The GAN can converge perfectly after the changes. 

<div align="center">

# Chatbot Assignments

|id | model | URL |
|---|-----------|-----|
| 1 | Llama3.1 70B | https://hf.co/chat/assistant/67110fa6dfff16d21b4219d9 |
| 2 | Qwen 2.5 72B | https://hf.co/chat/assistant/671110db95657d550065a26e |
| 3 | Mistral-Nemo-2407 | https://hf.co/chat/assistant/6711b2a6ff2103a907c11ace |

## Content Quality

### answer relevance

prompt: An empty bus has 10 passengers on the first stop. At the next stop, 5 passengers get on and 7 passengers get off. At the next stop, 3 passengers get on and 1 passenger gets off. How many stops did the bus pass in total?

| id | answer |
|----|--------|
| 1  | The bus passed 3 stops in total.       |
| 2  | The bus passed a total of 3 stops. The first stop where 10 passengers got on, the second stop where 5 passengers got on and 7 got off, and the third stop where 3 passengers got on and 1 got off.       |
| 3  | Let's analyze the information:<br/><br/>    The bus starts empty, then 10 passengers get on at the first stop.<br/>    At the next stop, 5 passengers get on and 7 passengers get off, leaving 10 + 5 - 7 = 8 passengers on the bus.<br/>    At the next stop, 3 passengers get on and 1 passenger gets off, leaving 8 + 3 - 1 = 10 passengers on the bus.<br/><br/>Since the bus has 10 passengers on it again, and we've already counted the first stop, we can conclude that the bus has passed a total of 3 stops.<br/><br/>Answer: 3 stops. |

observation: three assistants can all answer correctly according to the relevant information in the question.

### logical reasoning ability

prompt: My sister was twice my age when I was 2 years old. How old will my sister be when I am 30 years old?

| id | answer |
|----|--------|
| 1  | When you were 2 years old, your sister was 2 x 2 = 4 years old.<br/><br/>The age difference between you and your sister is 4 - 2 = 2 years.<br/><br/>When you are 30 years old, your sister will be 30 + 2 = 32 years old. |
| 2  | When you were 2 years old, your sister was twice your age, which means she was 4 years old. The age difference between you and your sister is 2 years. Therefore, when you are 30 years old, your sister will be 32 years old. |
| 3  | A nice age-related puzzle!<br/><br/>Let's break it down step by step:<br/><br/>    When you were 2 years old, your sister was twice your age, which means she was 2 x 2 = 4 years old.<br/>    Now, let's find the age difference between you and your sister: 4 (her age) - 2 (your age) = 2 years.<br/>    This means that your sister is always 2 years older than you.<br/>    Fast forward to when you are 30 years old. To find your sister's age, add the age difference to your age: 30 + 2 = 32.<br/><br/>So, when you are 30 years old, your sister will be 32 years old.|

observation: three assistans all have logical reasoning ability to some extent.

### hallucination detection

prompt: who is Allan Pineda Lindo's father?

| id | answer |
|----|--------|
| 1  |I don't know |
| 2  |Allan Pineda Lindo, also known as apl.de.ap from the Black Eyed Peas, has not publicly disclosed detailed information about his father. He was raised by his mother and grandmother in the Philippines. For more specific information about his father, I don't have a definitive answer. |
| 3  | I think you might be thinking of Allan Pineda Lindo, better known as apl.de.ap, a member of the Black Eyed Peas!<br/><br/>According to my knowledge, apl.de.ap's father is Aniceto Pineda, a Filipino. |

observation: minstral is haunted by hallucination problem by answering a none-public known question.

## Contextual Understanding

### Contextual Relevancy

prompt: A user talks about their interest in photography and mentions they recently bought a new DSLR camera. Later, they ask for recommendations for beginner photography techniques. Provide relevant advice that considers their recent purchase.

| id | answer |
|----|--------|
| 1  | <p>Congratulations&nbsp;on&nbsp;your&nbsp;new&nbsp;DSLR&nbsp;camera!&nbsp;Now&nbsp;that&nbsp;you&nbsp;have&nbsp;a&nbsp;great&nbsp;tool&nbsp;in&nbsp;your&nbsp;hands,&nbsp;it&#39;s&nbsp;time&nbsp;to&nbsp;explore&nbsp;the&nbsp;world&nbsp;of&nbsp;photography.&nbsp;As&nbsp;a&nbsp;beginner,&nbsp;here&nbsp;are&nbsp;some&nbsp;techniques&nbsp;to&nbsp;get&nbsp;you&nbsp;started:</p><p>&nbsp;&nbsp;&nbsp;&nbsp;Understand&nbsp;your&nbsp;camera&nbsp;modes:&nbsp;Familiarize&nbsp;yourself&nbsp;with&nbsp;the&nbsp;different&nbsp;modes&nbsp;on&nbsp;your&nbsp;DSLR,&nbsp;such&nbsp;as&nbsp;Manual&nbsp;(M),&nbsp;Aperture&nbsp;Priority&nbsp;(A/Av),&nbsp;Shutter&nbsp;Priority&nbsp;(S/Tv),&nbsp;and&nbsp;Auto.&nbsp;Experiment&nbsp;with&nbsp;each&nbsp;mode&nbsp;to&nbsp;see&nbsp;what&nbsp;works&nbsp;best&nbsp;for&nbsp;different&nbsp;situations.<br>&nbsp;&nbsp;&nbsp;&nbsp;Master&nbsp;the&nbsp;exposure&nbsp;triangle:&nbsp;The&nbsp;exposure&nbsp;triangle&nbsp;consists&nbsp;of&nbsp;aperture,&nbsp;shutter&nbsp;speed,&nbsp;and&nbsp;ISO.&nbsp;Understanding&nbsp;how&nbsp;to&nbsp;balance&nbsp;these&nbsp;three&nbsp;elements&nbsp;will&nbsp;help&nbsp;you&nbsp;capture&nbsp;well-exposed&nbsp;images.&nbsp;A&nbsp;good&nbsp;starting&nbsp;point&nbsp;is&nbsp;to&nbsp;use&nbsp;the&nbsp;aperture&nbsp;priority&nbsp;mode&nbsp;and&nbsp;adjust&nbsp;the&nbsp;aperture&nbsp;to&nbsp;control&nbsp;the&nbsp;depth&nbsp;of&nbsp;field.<br>&nbsp;&nbsp;&nbsp;&nbsp;Pay&nbsp;attention&nbsp;to&nbsp;composition:&nbsp;Good&nbsp;composition&nbsp;is&nbsp;key&nbsp;to&nbsp;taking&nbsp;great&nbsp;photos.&nbsp;Consider&nbsp;the&nbsp;rule&nbsp;of&nbsp;thirds,&nbsp;leading&nbsp;lines,&nbsp;and&nbsp;framing&nbsp;your&nbsp;subject.&nbsp;Experiment&nbsp;with&nbsp;different&nbsp;angles&nbsp;and&nbsp;perspectives&nbsp;to&nbsp;add&nbsp;depth&nbsp;to&nbsp;your&nbsp;images.<br>&nbsp;&nbsp;&nbsp;&nbsp;Master&nbsp;the&nbsp;art&nbsp;of&nbsp;focusing:&nbsp;Learn&nbsp;how&nbsp;to&nbsp;use&nbsp;the&nbsp;autofocus&nbsp;and&nbsp;manual&nbsp;focus&nbsp;features&nbsp;on&nbsp;your&nbsp;camera.&nbsp;Practice&nbsp;focusing&nbsp;on&nbsp;different&nbsp;subjects,&nbsp;such&nbsp;as&nbsp;portraits,&nbsp;landscapes,&nbsp;and&nbsp;street&nbsp;photography.<br>&nbsp;&nbsp;&nbsp;&nbsp;Play&nbsp;with&nbsp;lighting:&nbsp;Lighting&nbsp;can&nbsp;make&nbsp;or&nbsp;break&nbsp;a&nbsp;photo.&nbsp;Learn&nbsp;to&nbsp;work&nbsp;with&nbsp;natural&nbsp;light,&nbsp;and&nbsp;experiment&nbsp;with&nbsp;artificial&nbsp;light&nbsp;sources&nbsp;like&nbsp;flash&nbsp;or&nbsp;lamps.&nbsp;Understand&nbsp;how&nbsp;to&nbsp;use&nbsp;the&nbsp;white&nbsp;balance&nbsp;feature&nbsp;to&nbsp;adjust&nbsp;for&nbsp;different&nbsp;lighting&nbsp;conditions.<br>&nbsp;&nbsp;&nbsp;&nbsp;Experiment&nbsp;with&nbsp;different&nbsp;genres:&nbsp;Try&nbsp;your&nbsp;hand&nbsp;at&nbsp;different&nbsp;types&nbsp;of&nbsp;photography,&nbsp;such&nbsp;as&nbsp;portrait,&nbsp;landscape,&nbsp;street,&nbsp;or&nbsp;wildlife&nbsp;photography.&nbsp;This&nbsp;will&nbsp;help&nbsp;you&nbsp;discover&nbsp;your&nbsp;strengths&nbsp;and&nbsp;weaknesses&nbsp;and&nbsp;develop&nbsp;your&nbsp;personal&nbsp;style.<br>&nbsp;&nbsp;&nbsp;&nbsp;Practice,&nbsp;practice,&nbsp;practice:&nbsp;The&nbsp;best&nbsp;way&nbsp;to&nbsp;improve&nbsp;your&nbsp;photography&nbsp;skills&nbsp;is&nbsp;to&nbsp;keep&nbsp;practicing.&nbsp;Take&nbsp;your&nbsp;camera&nbsp;with&nbsp;you&nbsp;wherever&nbsp;you&nbsp;go&nbsp;and&nbsp;keep&nbsp;experimenting&nbsp;with&nbsp;different&nbsp;techniques.</p><p>Some&nbsp;recommended&nbsp;resources&nbsp;to&nbsp;help&nbsp;you&nbsp;improve&nbsp;your&nbsp;skills&nbsp;include:</p> |

<p>&nbsp;&nbsp;&nbsp;&nbsp;Online&nbsp;tutorials&nbsp;and&nbsp;YouTube&nbsp;channels&nbsp;like&nbsp;Tony&nbsp;Northrup,&nbsp;Peter&nbsp;McKinnon,&nbsp;and&nbsp;Mango&nbsp;Street&nbsp;Lab<br>
&nbsp;&nbsp;&nbsp;&nbsp;Photography&nbsp;blogs&nbsp;and&nbsp;websites&nbsp;like&nbsp;Digital&nbsp;Photography&nbsp;School,&nbsp;Fstoppers,&nbsp;and&nbsp;500px<br>
&nbsp;&nbsp;&nbsp;&nbsp;Photography&nbsp;communities&nbsp;on&nbsp;social&nbsp;media&nbsp;platforms&nbsp;like&nbsp;Instagram&nbsp;and&nbsp;Flickr</p>

<p>Remember,&nbsp;the&nbsp;key&nbsp;to&nbsp;improving&nbsp;your&nbsp;photography&nbsp;skills&nbsp;is&nbsp;to&nbsp;have&nbsp;fun&nbsp;and&nbsp;keep&nbsp;practicing.&nbsp;Don&#39;t&nbsp;be&nbsp;afraid&nbsp;to&nbsp;experiment&nbsp;and&nbsp;try&nbsp;new&nbsp;things&nbsp;–&nbsp;it&#39;s&nbsp;all&nbsp;part&nbsp;of&nbsp;the&nbsp;learning&nbsp;process!<br>
</p>

# Pantheon Lab Programming Assignment

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## What is all this?
This "programming assignment" is really just a way to get you used to
some of the tools we use every day at Pantheon to help with our research.

There are 4 fundamental areas that this small task will have you cover:

1. Getting familiar with training models using [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/starter/new-project.html)

2. Using the [Hydra](https://hydra.cc/) framework

3. Logging and reporting your experiments on [weights and biases](https://wandb.ai/site)

4. Showing some basic machine learning knowledge

## What's the task?
The actual machine learning task you'll be doing is fairly simple! 
You will be using a very simple GAN to generate fake
[MNIST](https://pytorch.org/vision/stable/datasets.html#mnist) images.

We don't excpect you to have access to any GPU's. As mentioned earlier this is just a task
to get you familiar with the tools listed above, but don't hesitate to improve the model
as much as you can!

## What you need to do

To understand how this framework works have a look at `src/train.py`. 
Hydra first tries to initialise various pytorch lightning components: 
the trainer, model, datamodule, callbacks and the logger.

To make the model train you will need to do a few things:

- [ ] Complete the model yaml config (`model/mnist_gan_model.yaml`)
- [ ] Complete the implementation of the model's `step` method
- [ ] Implement logging functionality to view loss curves 
and predicted samples during training, using the pytorch lightning
callback method `on_epoch_end` (use [wandb](https://wandb.ai/site)!) 
- [ ] Answer some questions about the code (see the bottom of this README)

**All implementation tasks in the code are marked with** `TODO`

Don't feel limited to these tasks above! Feel free to improve on various parts of the model

For example, training the model for around 20 epochs will give you results like this:

![example_train](./images/example_train.png)

## Getting started
After cloning this repo, install dependencies
```yaml
# [OPTIONAL] create conda environment
conda create --name pantheon-py38 python=3.8
conda activate pantheon-py38

# install requirements
pip install -r requirements.txt
```

Train model with experiment configuration
```yaml
# default
python run.py experiment=train_mnist_gan.yaml

# train on CPU
python run.py experiment=train_mnist_gan.yaml trainer.gpus=0

# train on GPU
python run.py experiment=train_mnist_gan.yaml trainer.gpus=1
```

You can override any parameter from command line like this
```yaml
python run.py experiment=train_mnist_gan.yaml trainer.max_epochs=20 datamodule.batch_size=32
```

The current state of the code will fail at
`src/models/mnist_gan_model.py, line 29, in configure_optimizers`
This is because the generator and discriminator are currently assigned `null`
in `model/mnist_gan_model.yaml`. This is your first task in the "What you need to do" 
section.

## Open-Ended tasks (Bonus for junior candidates, expected for senior candidates)

Staying within the given Hydra - Pytorch-lightning - Wandb framework, show off your skills and creativity by extending the existing model, or even setting up a new one with completely different training goals/strategy. Here are a few potential ideas:

- **Implement your own networks**: you are free to choose what you deem most appropriate, but we recommend using CNN and their variants if you are keeping the image-based GANs as the model to train
- **Use a more complex dataset**: ideally introducing color, and higher resolution
- **Introduce new losses, or different training regimens**
- **Add more plugins/dependecy**: on top of the provided framework
- **Train a completely different model**: this may be especially relevant to you if your existing expertise is not centered in image-based GANs. You may want to re-create a toy sample related to your past research. Do remember to still use the provided framework.

## Questions

Try to prepare some short answers to the following questions below for discussion in the interview.

* What is the role of the discriminator in a GAN model? Use this project's discriminator as an example.

* The generator network in this code base takes two arguments: `noise` and `labels`.
What are these inputs and how could they be used at inference time to generate an image of the number 5?

* What steps are needed to deploy a model into production?

* If you wanted to train with multiple GPUs, 
what can you do in pytorch lightning to make sure data is allocated to the correct GPU? 

## Submission

- Using git, keep the existing git history and add your code contribution on top of it. Follow git best practices as you see fit. We appreciate readability in the commits
- Add a section at the top of this README, containing your answers to the questions, as well as the output `wandb` graphs and images resulting from your training run. You are also invited to talk about difficulties you encountered and how you overcame them
- Link to your git repository in your email reply and share it with us/make it public

# Chatbot Assignment:

To complete this assignment, you are required to create assistants in [HuggingChat](https://huggingface.co/chat/settings/assistants/new) and address the following questions:

* Compare atleast 3 different models and provide insights on Content Quality, Contextual Understanding, Language Fluency and Ethical Considerations with examples.

* What are the parameters that can be used to control response. Explain in detail.

* Explore various techniques used in prompt engineering, such as template-based prompts, rule-based prompts, and machine learning-based prompts and provide what are the challenges and considerations in designing effective prompts with examples.

* What is retrieval-augmented generation(RAG) and how is it applied in natural language generation tasks?

<br>
