from model import Model

def train_all():
    # we call the Model class to impliment the model
    model = Model()
    model.create_model()
    model.train_model('train')
    model.save_model()
