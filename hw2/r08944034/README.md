## Train step
(batch_size=9 optimizer=nn.AdamW loss_func=CrossEntropy)


```python Train-Context.py```

## question5 implement
#### it will read the train.json and calculate the sum of result length

```python Q5.py```
```
plt.hist(len_list,bins=15,range=(0,100),normed=True ,cumulative=True,histtype=u'bar',rwidth=0.5)
```

## question6 implement
#### result01 ~ result09 represent the result of threshold 0.1~0.9 it will read and plot.
```python Q6.py```

```
plt.plot(x,F1_o,'o-',color = 'r', label="overall")
plt.plot(x,F1_u,'o-',color = 'g', label="unanswerable")
plt.plot(x,F1_a,'o-',color = 'b', label="answerable")
plt.plot(x,EM_o,'o-',color = 'r', label="overall")
plt.plot(x,EM_u,'o-',color = 'g', label="unanswerable")
plt.plot(x,EM_a,'o-',color = 'b', label="answerable")
```


