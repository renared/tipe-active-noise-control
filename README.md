# Open loop active noise control
TIPE 2019 : Contrôle actif du bruit acoustique

See presentation (TIPE19-03.pdf) (in french).

The code was written for a presentation, so it is not really practical, but some functions can be useful :
* Overlap-add filtering: filter_OLA
* Estimate transfer function between two pure sine signals : transfer_sine
* Real-time audio filtering using sounddevice for audio IO: engine* functions (not a very appropriate name)

## References

[1] Emmanuel Friot : Une introduction au contrôle acoustique actif. : DEA. 2006. <cel-
00092972>
  
[2] Gérard MANGIANTE : Contrôle actif des bruits - Bases théoriques : Techniques de
l'ingénieur. 2008.
