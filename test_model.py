import torch
import torch.nn as nn

# Load the model from a .pt file
path = "/home/parallels/pupper_v3_1_ws/src/neural_controller/policy_lstm_1.pt"

# BE EXTREMELY CAREFUL ABOUT HIDDEN STATE REFERENCES AND CHANGES
# use model_verb_cpu.reset_memory() to set them to zero
model_verb_cpu = torch.load(path, map_location="cpu").eval()

od1 = model_verb_cpu.state_dict()


class MyLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, dense_sizes, output_size):
        super(MyLSTMModel, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)

        # Three dense layers with ELU activation
        self.fc1 = nn.Linear(hidden_size, dense_sizes[0])
        self.elu1 = nn.ELU()
        self.fc2 = nn.Linear(dense_sizes[0], dense_sizes[1])
        self.elu2 = nn.ELU()
        self.fc3 = nn.Linear(dense_sizes[1], dense_sizes[2])
        self.elu3 = nn.ELU()

        # Final dense layer with linear activation
        self.fc4 = nn.Linear(dense_sizes[2], output_size)

        self.hidden_size = hidden_size

    def forward(self, x, hidden_and_cell_state=None):
        # Forward pass through LSTM layer
        if hidden_and_cell_state is not None:
            out, (h, c) = self.lstm(
                x, (hidden_and_cell_state[0], hidden_and_cell_state[1])
            )
        else:
            out, (h, c) = self.lstm(x)

        # Flatten the output if needed
        out = out[:, -1, :]  # Select the last time step's output

        # Forward pass through dense layers with ELU activation
        out = self.elu1(self.fc1(out))
        out = self.elu2(self.fc2(out))
        out = self.elu3(self.fc3(out))

        # Forward pass through the final dense layer with linear activation
        out = self.fc4(out)

        return out, (h, c)


# Define the input size, hidden size, number of LSTM layers, and output size
input_size = 48
hidden_size = 512
output_size = 12  # You can adjust this based on your task
dense_sizes = [512, 256, 128]

# Create an instance of the model
model = MyLSTMModel(input_size, hidden_size, dense_sizes, output_size)


# model_output, _ = model(random_input.unsqueeze(0))
# print("model out with unset hidden/cell and weights/biases:\n", model_output)


model.fc1.weight = nn.Parameter(od1["actor.0.weight"])
model.fc1.bias = nn.Parameter(od1["actor.0.bias"])

model.fc2.weight = nn.Parameter(od1["actor.2.weight"])
model.fc2.bias = nn.Parameter(od1["actor.2.bias"])

model.fc3.weight = nn.Parameter(od1["actor.4.weight"])
model.fc3.bias = nn.Parameter(od1["actor.4.bias"])

model.fc4.weight = nn.Parameter(od1["actor.6.weight"])
model.fc4.bias = nn.Parameter(od1["actor.6.bias"])

model.lstm.weight_hh_l0 = nn.Parameter(od1["memory.weight_hh_l0"])
model.lstm.bias_hh_l0 = nn.Parameter(od1["memory.bias_hh_l0"])

model.lstm.weight_ih_l0 = nn.Parameter(od1["memory.weight_ih_l0"])
model.lstm.bias_ih_l0 = nn.Parameter(od1["memory.bias_ih_l0"])

random_input = torch.zeros(1, 48)  # torch.randn(1, 48)

hidden_state = model_verb_cpu.hidden_state  # reference not copy
cell_state = model_verb_cpu.cell_state  # reference not copy

hope_works_out, (h1, c1) = model(random_input.unsqueeze(0), (hidden_state, cell_state))
print(
    "model output1 with set hidden/celland and weights/biases:\n",
    hope_works_out,
)

print("JIT output1: ", model_verb_cpu(random_input))

hope_works_out, (h1, c1) = model(random_input.unsqueeze(0), (h1, c1))
print(
    "model output2 with set hidden/celland and weights/biases:\n",
    hope_works_out,
)

print("JIT output2: ", model_verb_cpu(random_input))


# Print the model's architecture
print(model)
