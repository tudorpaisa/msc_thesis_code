# Cruft essay to give me an idea how to do my thesis presentation

Hello everybody and welcome to my presentation.

<TopicOfTheDay>

With their reemergence, Neural Networks have brought new ways to tackle ML problems as well as possibilities to further artificial intelligence research. Deep learning as a field promises to discover rich hierarchical models over various AI applications such as speech recognition, image classification, text generation, etc. The majority of developments usually involve discriminative models where high dimensional input is - for instance - assigned a class label. That being said, as you can probably guess, this is more that it meets the eye.

Two deep learning task that have recently gathered a lot of attention are generative models and few-shot learning, both of which are central to my research. As such, it's necessary to briefly explain each of them before I can expand on the topic of my project.

Generative models are somewhat self-explanatory: a NN that is able to generate data. As a concept it's easy to grasp its purpose however, developments in generative models have been fairly dull until fairly recently. A major breakthrough in this task was the discovery of Generative Adversarial Networks by Goodfellow et. al. (2014), and the more modern adaptations of a Long Short-Term Memory Network by Grave (????). That being said, most experiments have been in the field of computer vision, natural language, and lastly arts, with music being one of the least explored areas. I will come back to this point later. After I tell you more about the second part of my thesis.

Now, it is well known that Neural Networks tend to be "data-hungry", something which is due to its optimization algorithm, Stochastic Gradient Descent. The iterative nature of the algorithm does not allow the network to converge in a limited number of steps. Solving this issue would obviously make DL models easier to train however, there are other benefits as well. Recall that humans can generalize after one or few examples of a given object, something which DL algorithms fail spectacularly at doing. Bridging the gap between how humans learn and how neural networks learn can alleviate the constant need for large datasets to ensure reasonable performances in deep learning applications. This is the few-shot learning problem: finding ways to train a Neural Network so it is able to work with small amounts of data.

I hope you can see where this is going. The overwhelming majority of few-shot generative developments have mostly been in the field of computer vision, followed by natural language, thus leaving the musical domain largely unexplored. As such, for my research I will adapt two generative models to the few-shot learning problem and evaluate their performances when it comes to generating music.

The models in question are C-RNN-GAN, a generative adversarial network with two LSTM layers with 350 hidden units each for both the generator and discriminator. The only difference between the generator and the discriminator is that the latter has a bidirectional. The second network I will adapt is PerformanceRNN, a three layer unidirectional LSTM network with 512 units each.

To make the networks work with a limited dataset, I will be making use of the Reptile algorithm; a recent state of the art solution to the few-shot learning problem. It counteracts the shortcomings of gradient-optimization algorithms by learning the model's initial parameters in order to maximize its performance on novel tasks.

I will be comparing the performance of these two models to this baseline: a network with one LSTM layer and 200 hidden units trained on the entire dataset. Additionally I will be comparing the performance of the generative models with properties of the dataset itself.

Which leads me to another important aspect of my research: the dataset. I will be using the MAESTRO dataset. It contains 1,184 piano performances recorded as MIDI files from 9 years of the International Piano-e-Competition. The performances consist of 430 individual compositions, 6.18 million notes played and around 172 hours of playback.

The evaluation of the models will use negative log likelihood (as it's the standard evaluation metric) along with the number of statistically different bins and several domain-specific metrics: polyphony (how often a minimum of two notes are played simultaneously), scale consistency (fraction of notes that are part in a standard scale), repetitions (the number of consecutive note subsequences), and tone span (the difference in semitones between the lowest and highest note.

Finally, these are the milestones I allocated to myself. It should be noted that I already wrote the code for the one-hot encoder, the baseline and the reptile learning algorithm.
