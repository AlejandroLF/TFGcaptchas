import time
start = time.time()

from create_captcha import create_captcha, np
from segment_image import segment_image

from sklearn.utils import check_random_state
random_state = check_random_state(14)
letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
shear_values = np.arange(0, 0.5, 0.05)

def generate_sample(random_state=None):
    random_state = check_random_state(random_state)
    letter = random_state.choice(letters)
    shear = random_state.choice(shear_values)

    return create_captcha(letter, shear=shear, size=(20,20)), letters.index(letter)

dataset, targets = zip(*(generate_sample(random_state) for _ in range(10000)))
dataset = np.array(dataset)
targets = np.array(targets)

from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(categories='auto')
y = onehot.fit_transform(targets.reshape(targets.shape[0], 1))
y = y.todense()


from skimage.transform import resize

dataset = np.array([resize(segment_image(sample)[0], (20, 20)) for sample in dataset])
X = dataset.reshape((dataset.shape[0], dataset.shape[1] * dataset.shape[2]))  # Flatten images

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)


from pybrain.datasets import SupervisedDataSet

training = SupervisedDataSet(X.shape[1], y.shape[1])
for i in range(X_train.shape[0]):
    training.addSample(X_train[i], y_train[i])

testing = SupervisedDataSet(X.shape[1], y.shape[1])
for i in range(X_test.shape[0]):
    testing.addSample(X_test[i], y_test[i])


from pybrain.tools.shortcuts import buildNetwork
net = buildNetwork(X.shape[1], 100, y.shape[1], bias=True)

from pybrain.supervised.trainers import BackpropTrainer
trainer = BackpropTrainer(net, training, learningrate=0.01, weightdecay=0.01)
start2 = time.time()
trainer.trainEpochs(epochs=20)
end2 = time.time()

predictions = trainer.testOnClassData(dataset=testing)
from sklearn.metrics import f1_score
print("F-score: {0:.2f}".format(f1_score(predictions, y_test.argmax(axis=1), average="micro")))


def predict_captcha(captcha_image, neural_network):
    subimages = segment_image(captcha_image)
    predicted_word = ""
    for subimage in subimages:
        subimage = resize(subimage, (20, 20))
        outputs = net.activate(subimage.flatten())
        prediction = np.argmax(outputs)
        predicted_word += letters[prediction]
    return predicted_word

def test_prediction(word, net, shear=0.2):
    captcha = create_captcha(word, shear=shear)
    prediction = predict_captcha(captcha, net)
    # prediction = prediction[:4]
    return word == prediction, word, prediction

#print(test_prediction("GENE", net))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), predictions)
number_of_letter_predictions = []
for i in range(len(cm)):
    predictions = 0
    for j in range(len(cm)):
        predictions += cm[j][i]
    number_of_letter_predictions.append(predictions)

def confusion_probability(original_letter, new_letter):
    original_letter_index = letters.index(original_letter)
    new_letter_index = letters.index(new_letter)
    original_letter_predictions = number_of_letter_predictions[original_letter_index]
    # No info at all in this case
    if original_letter_predictions == 0:
        return 0.5
    return cm[new_letter_index][original_letter_index] / original_letter_predictions

def compute_distance(prediction, word):
    return len(prediction) - sum(confusion_probability(prediction[i], word[i]) for i in range(len(prediction)))


from operator import itemgetter
def improved_prediction(captcha, net, dictionary):
    lessThan4 = False
    prediction = predict_captcha(captcha, net)
    if len(prediction) < 4:
        lessThan4 = True
    if prediction not in dictionary:
        distances = sorted([(word, compute_distance(prediction, word)) for word in dictionary], key=itemgetter(1))
        best_word = distances[0]
        prediction = best_word[0]
    return prediction, lessThan4

def test_improved_prediction(word, net, dictionary, shear=0.2):
    lessThan4 = False
    captcha = create_captcha(word, shear=shear)
    prediction, lessThan4 = improved_prediction(captcha, net, dictionary)
    return word == prediction, word, prediction, lessThan4

#import nltk
#nltk.download('words')
from nltk.corpus import words

valid_words = [word.upper() for word in words.words() if len(word) == 4]
num_correct = 0
num_incorrect = 0
num_lt4c = 0
num_lt4i = 0
for word in valid_words:
    correct, word, prediction, lessThan4 = test_improved_prediction(word, net, valid_words, shear=random_state.choice(shear_values))
    if correct:
        num_correct += 1
    else:
        num_incorrect += 1

print("Number of correct is {0}".format(num_correct))
print("Number of incorrect is {0}".format(num_incorrect))
print("Success rate: {0} %".format(num_correct / (num_correct + num_incorrect) * 100))

end = time.time()
print("Total time: {0}\nTraining time: {1}".format(end - start, end2 - start2))

########################################