import torch 
import torch.nn as nn 
import torch.optim as optim 
from torchvision import datasets, transforms 
from torch.autograd import Variable 
# Training settings 
batch_size    = 64 
num_classes   = 10
learningrate  = 0.01 
momentum      = 0.5
num_epochs    = 2

#cnn paras:
cnn_in_channels   = 1    
cnn_out_channels1 = 10
cnn_out_channels2 = 20   #the second cnn layers out_channels
cnn_kernel_size   = 5

#cnn-rnn paras:
cr_in_channels   = 1
cr_out_channels  = 10
cr_kernel_size   = 5
cr_num_layers    = 1
cr_input_size    = 144
cr_hidden_size   = 128

#rnn paras:
rnn_sequence_length = 28 
rnn_input_size      = 28  
rnn_hidden_size     = 128  
rnn_num_layers      = 1 

# MNIST Dataset 
train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True) 
test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor()) 

# Data Loader (Input Pipeline) 

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True) 
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

class CNN(nn.Module):
    def __init__(self,in_channels,out_channels1,out_channels2,kernel_size,num_classes):
        super(CNN, self).__init__()

        self.in_channels   = in_channels
        self.out_channels1 = out_channels1
        self.out_channels2 = out_channels2
        self.kernel_size   = kernel_size
        self.num_classes   = num_classes
         
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels1, self.kernel_size)
        self.conv2 = nn.Conv2d(self.out_channels1, self.out_channels2, self.kernel_size)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, self.num_classes)# fully connect

    def forward(self, x):
        in_size = x.size(0)          # one batch
        x = self.mp(self.conv1(x))   #input:64*1*28*28, output:64*10*12*12
        x = self.mp(self.conv2(x))   #input:64*10*12*12,output:64*20*4*4
        x = x.view(in_size, -1)      #flatten the tensor,input:64*20*4*4,output:64*320
        x = self.fc(x)               #input:64*320,output:64*10
        return x

class CNNGRU(nn.Module):
    def __init__(self,in_channels,out_channels, kernel_size,num_layers,input_size,hidden_size,num_classes):
        super(CNNGRU, self).__init__()
        
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.num_classes  = num_classes
        
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size)
        self.mp   = nn.MaxPool2d(2)
        self.gru  = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        # fully connect
        self.fc   = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        in_size = x.size(0) # one batch
        x       = self.mp(self.conv(x))
        x       = x.view(in_size,10,-1)
        h0      = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        out, _  = self.gru(x, h0)
        out     = self.fc(out[:, -1, :]) # flatten the tensor
        return out

class CNNBiGRU(nn.Module):
    def __init__(self,in_channels,out_channels, kernel_size,num_layers,input_size,hidden_size,num_classes):
        super(CNNBiGRU, self).__init__()
                
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.num_classes  = num_classes
        
        self.conv  = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size)
        self.mp    = nn.MaxPool2d(2)
        self.bigru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True,bidirectional = True)
        # fully connect
        self.fc    = nn.Linear(self.hidden_size*2, self.num_classes)

    def forward(self, x):
        in_size = x.size(0) # one batch
        x       = self.mp(self.conv(x))
        x       = x.view(in_size,10,-1)
        h0      = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))
        out, _  = self.bigru(x, h0)
        out     = self.fc(out[:, -1, :]) # flatten the tensor
        return out
class CNNLSTM(nn.Module):
    def __init__(self,in_channels,out_channels, kernel_size,num_layers,input_size,hidden_size,num_classes):
        super(CNNLSTM, self).__init__()
        
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.num_classes  = num_classes
        
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size)
        self.mp   = nn.MaxPool2d(2)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc   = nn.Linear(self.hidden_size, self.num_classes)       # fully connect

    def forward(self, x):
        in_size = x.size(0) # one batch:64
        x       = self.mp(self.conv(x))#input:64*1*28*28,conv_out:64*10*24*24,out:pool_out:64*10*12*12
        x       = x.view(in_size,10,-1)#input:64*10*12*12 ,out:64*10*144
        h0      = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c0      = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        out, _  = self.lstm(x, (h0, c0))#input:64*10*144 ,out:64*10*128
        out     = self.fc(out[:, -1, :]) # flatten the tensor,#input:64*10*128,out[:, -1, :]:64*128,out:64*10
        return out
class CNNBiLSTM(nn.Module):
    def __init__(self,in_channels,out_channels, kernel_size,num_layers,input_size,hidden_size,num_classes):
        super(CNNBiLSTM, self).__init__()
                        
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.num_classes  = num_classes
        
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size)
        self.mp   = nn.MaxPool2d(2)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True,bidirectional = True)
        self.fc   = nn.Linear(self.hidden_size*2, self.num_classes)# fully connect

    def forward(self, x):
        in_size = x.size(0) # one batch
        x       = self.mp(self.conv(x))
        x       = x.view(in_size,10,-1)
        h0      = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))
        c0      = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))
        out, _  = self.lstm(x, (h0, c0))
        out     = self.fc(out[:, -1, :]) # flatten the tensor
        return out
class GRU(nn.Module):
    def __init__(self, sequence_length, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()

        self.sequence_length = sequence_length
        self.hidden_size     = hidden_size
        self.num_layers      = num_layers
        self.input_size      = input_size
        self.num_classes     = num_classes

        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True) 
        self.fc = nn.Linear(self.hidden_size, self.num_classes)
    
    def forward(self, x):
        h0       = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) 
        x        = x.view(-1, self.sequence_length, self.input_size)
        out, h_n = self.gru(x, h0)  
        out      = self.fc(out[:, -1, :])   
        return out

class BiGRU(nn.Module):
    def __init__(self, sequence_length, input_size, hidden_size, num_layers, num_classes):
        super(BiGRU, self).__init__()

        self.sequence_length = sequence_length
        self.hidden_size     = hidden_size
        self.num_layers      = num_layers
        self.input_size      = input_size
        self.num_classes     = num_classes

        self.bigru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True,bidirectional = True) 
        self.fc = nn.Linear(self.hidden_size*2, self.num_classes)
    
    def forward(self, x):
        h0       = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)) 
        x        = x.view(-1, self.sequence_length, self.input_size)
        out, h_n = self.bigru(x, h0)
        out      = self.fc(out[:, -1, :]) 
        return out

class LSTM(nn.Module):
    def __init__(self, sequence_length, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        
        self.sequence_length = sequence_length
        self.hidden_size     = hidden_size
        self.num_layers      = num_layers
        self.input_size      = input_size
        self.num_classes     = num_classes

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True) 
        self.fc   = nn.Linear(self.hidden_size, self.num_classes)
    
    def forward(self, x):
        h0              = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) 
        c0              = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        x               = x.view(-1, self.sequence_length, self.input_size)# input: 64*1*28*28, out:64*28*28
        out, (h_n, c_n) = self.lstm(x, (h0, c0))#input:64*28*28,out:64*28*128    
        out             = self.fc(out[:, -1, :]) #input:64*28*128 , out[:, -1, :]:64*128,out:64*10  
        return out

class BiLSTM(nn.Module):
    def __init__(self,  sequence_length, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()

        self.sequence_length = sequence_length
        self.hidden_size     = hidden_size
        self.num_layers      = num_layers
        self.input_size      = input_size
        self.num_classes     = num_classes

        self.bilstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True,bidirectional = True) 
        self.fc = nn.Linear(self.hidden_size*2, self.num_classes)
    
    def forward(self, x):
        h0              = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)) 
        c0              = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))
        x               = x.view(-1, self.sequence_length, self.input_size)
        out, (h_n, c_n) = self.bilstm(x, (h0, c0)) 
        out             = self.fc(out[:, -1, :])  
        return out


test_model = ['CNN','CNNGRU','CNNBiGRU','CNNLSTM','CNNBiLSTM','GRU','BiGRU','LSTM','BiLSTM']
classifiers = {'CNN'      :CNN(cnn_in_channels ,cnn_out_channels1 ,cnn_out_channels2 ,cnn_kernel_size,num_classes),
               'CNNGRU'   :CNNGRU(cr_in_channels,cr_out_channels, cr_kernel_size,cr_num_layers,cr_input_size,cr_hidden_size,num_classes),
               'CNNBiGRU' :CNNBiGRU(cr_in_channels,cr_out_channels, cr_kernel_size,cr_num_layers,cr_input_size,cr_hidden_size,num_classes),
               'CNNLSTM'  :CNNLSTM(cr_in_channels,cr_out_channels, cr_kernel_size,cr_num_layers,cr_input_size,cr_hidden_size,num_classes),
               'CNNBiLSTM':CNNBiLSTM(cr_in_channels,cr_out_channels, cr_kernel_size,cr_num_layers,cr_input_size,cr_hidden_size,num_classes),
               'GRU'      :GRU(rnn_sequence_length,rnn_input_size, rnn_hidden_size, rnn_num_layers, num_classes),
               'BiGRU'    :BiGRU(rnn_sequence_length,rnn_input_size, rnn_hidden_size, rnn_num_layers, num_classes),
               'LSTM'     :LSTM(rnn_sequence_length,rnn_input_size, rnn_hidden_size, rnn_num_layers, num_classes),
               'BiLSTM'   :BiLSTM(rnn_sequence_length,rnn_input_size, rnn_hidden_size, rnn_num_layers, num_classes),
}

#Train the model
for classifier in test_model:
    print('*********** %s ************' %classifier)
    model = classifiers[classifier]
    optimizer = optim.SGD(model.parameters(), learningrate, momentum)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)  
            labels = Variable(labels)
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            if (i+1) % 300 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

# Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()

    print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))










