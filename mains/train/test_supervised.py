from mains.train.train_supervised import train_supervised
from learning.meters_and_metrics.meter_server import get_current_meters
from pprint import pprint


if __name__ == "__main__":
    test_loss, train_loss = train_supervised(test_only=True)
    print("Results:")
    pprint(get_current_meters())
    print(f"Test loss: {test_loss}; Train loss: {train_loss}")
