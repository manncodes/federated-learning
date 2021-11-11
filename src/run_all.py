import os

NUM_CLIENTS = 4

if __name__ == "__main__":
    # change to current directory to run the script
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # run server.py in another cmd window
    os.system("start cmd /K python server.py")

    # run client.py in another cmd windows
    for client_id in range(NUM_CLIENTS):
        # to run default cifar clients FedAvg
        os.system(f"start cmd /K python client-basic.py {client_id}")

        # # to run multiple clients[Custom FedBN]
        # os.system(f"start cmd /K python client.py {client_id}")
