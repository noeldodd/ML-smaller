# ML-smaller
Shrinkify a trainable ML network for a single-zone thermostat example

- 4 inputs
- 1 output

The math in this version ('smaller', not 'smallest) is still float, but assume the notes below on using byte-friendly values.

## Inputs

The four nodes represent Day of Week, Time of Day, Exterior Temp, and Interior Temp. All the target values are intended to fit (eventually) in 8 bits, so are calculated for testing as fractions of 255.

- Day of week; use decimals like:

1. 0: 0.0 (unused / invalid). Suspect starting with weekdays first will be more stable:
2. 31: ~0.1216 Mon
3. 63: ~0.2471 Tue
4. 95: ~0.3725 Wed
5. 127: ~0.4980 Thur
6. 159: ~0.6235 Fri
7. 191: ~0.7490 Sat
8. 223: ~0.8745 Sun

- Time of Day; Fixed point decimals where the first two digits are hour, and the 'tenths' place is 1/10 of an hour. The 24 hours are:

1. 000/255 = 0.0000
2. 010/255 = ~0.0392
3. 020/255 = ~0.0784
4. 030/255 = ~0.1176
5. 040/255 = ~0.1569
6. 050/255 = ~0.1961
7. 060/255 = ~0.2353
8. 070/255 = ~0.2745
9. 080/255 = ~0.3137
10. 090/255 = ~0.3529
11. 100/255 = ~0.3922
12. 110/255 = ~0.4314
13. 120/255 = ~0.4706
14. 130/255 = ~0.5098
15. 140/255 = ~0.5490
16. 150/255 = ~0.5882
17. 160/255 = ~0.6275
18. 170/255 = ~0.6667
19. 180/255 = ~0.7059
20. 190/255 = ~0.7451
21. 200/255 = ~0.7843
22. 210/255 = ~0.8235
23. 220/255 = ~0.8627
24. 230/255 = ~0.9020
