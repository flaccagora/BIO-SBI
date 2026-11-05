
# Experiments

EXP: x modello (GW - BH) x inferenza (SBI) x seed (di inferenza e dataset) X combinazioni di errori

- **Morte cellulare**:
    - tasso di morte degli organoidi influenza soltanto le condinzioni iniziali (distribuzione di partenza numero e tipo), non la dinamica di crescita.
        1a EXP: quanto è stabile l'inferenza al variare delle condizioni iniziali?
    - tasso di morte per tipo cellulare:
        influenza la dinamica di crescita, ma non le condizioni iniziali?
        1b EXP: quanto i tassi di morte influenzano il numero di organoidi "osservati"
        1c EXP: cosa succede se fisso tassi di morte e inferisco gli altri parametri al variare dei tassi di morte? (per tipo cellulare (death_t1, death_t2))? Se le soluzioni sono infinite -> serve un modo per stimare la morte (marker?)

- **Tipizzazione**:
    - nel clustering è presente un "terzo tipo" (doppio negativo) che non è presenta nel modello ma che vengono classificate come tipo 1 o tipo 2. Questo introduce un errore di tipizzazione che può essere quantificato.
        2a EXP: quanto l'errore di tipizzazione influenza l'inferenza dei parametri?
        2b EXP: come si comporta l'inferenza quando si introduce un errore di tipizzazione?
        2c EXP: cosa succede se inserisco un terzo tipo? (vincolato da trajectory inference)

    /Biologico il terzo tipo sta andando solo in differentiated (traj inf)? 
    /Biologico percentuale di ++ ~5% -- ~15%
    /Biologico quanto mislabelling from 2D to 3D?

- **Errori di misura**:
    - Tempi di divisione e desincronizzazione (per tipo? la stima è globale)
        EXP: quantificazione errore nel caso di tempo di divisione differenti per tipo 
        EXP: Inattivazione agisce direttamente su questo
    - 
- **#data**
    - benchmark how much data is needed to infer parameters with a given accuracy.