## Train steps
### extractive

```
python3.7 model1_train.py
```
### seq2seq
```
python3.7 model2_train.py
```
### seq2seq2 with attention
```
python3.7 model3_train.py
```

## How to plot the figures

### implement step
```
ax = fig.add_subplot(111)
cax = ax.matshow(attn_matrix.detach().numpy(), cmap='bone')
fig.colorbar(cax)
    # Set up axes
#ax.set_xticklabels(valid_data+[''], rotation=90)
#ax.set_yticklabels(out_words+[''])
ax.set_xticklabels(['']+out_words)
ax.set_yticklabels(['']+valid_data, rotation=90)
    # Show label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()
```
### how to run
```
python3.7 visual_attention.py
```