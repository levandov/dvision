# dvision comp vision project
# Problem Formulation:
There are 30 satellite pictures of houses and 25 corresponding labels that indicate the roofs. Take those 25 data points and train a neural network on them - you are completely free about the architecture and are of course allowed to use any predefined version of networks, however, you should be able to explain what you are doing - in terms of code as well as in terms of why certain steps are good choices. The preferred language is Python, but you can also use other languages. Please evaluate your network on the 5 remaining test images by making predictions of the roofs - send us the predictions and ideally some comments on what you have been doing. Everything else we will discuss from there. The data can be found at images/ resp. labels/ directories

#Solution:
I decided to use very novel dinov3 from facebookresearch (fork from their github facebookresearch/dinov3),
while I read about dino2 before. The result is the jupyter notebook
dvision.ipynb
as well as its conversion to the text python file
dvision.py
The resulted model itself is exported into
dvision.ipynb
