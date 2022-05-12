from glob import glob
from config import config
from box import Box
config=Box(config)
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

path = glob(f'./{config.model.name}/default/version_0/events*')[0]
event_acc = EventAccumulator(path, size_guidance={'scalars': 0})
event_acc.Reload()

scalars = {}
for tag in event_acc.Tags()['scalars']:
    events = event_acc.Scalars(tag)
    scalars[tag] = [event.value for event in events]

import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
print(scalars)
plt.plot(range(len(scalars['lr-AdamW'])), scalars['lr-AdamW'])
plt.xlabel('epoch')
plt.ylabel('lr')
plt.title('adamw lr')

plt.subplot(1, 2, 2)
plt.plot(range(len(scalars['train_loss'])), scalars['train_loss'], label='train_loss')
plt.plot(range(len(scalars['val_loss'])), scalars['val_loss'], label='val_loss')
plt.legend()
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.title('train/val rmse')
plt.show()