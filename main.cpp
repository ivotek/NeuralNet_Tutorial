#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ctime>
#include <cstdlib>
#include "NeuralNetwork.h"

// ---------------------------------------------------------------------
// SCALE: costante di normalizzazione
// ---------------------------------------------------------------------
// La rete usa la funzione sigmoide come attivazione, il cui output è
// sempre compreso tra 0 e 1. Per questo motivo non le si possono dare
// in pasto numeri "grezzi" come 20 e 30 (troppo grandi): li dividiamo
// per SCALE prima di passarli alla rete (normalizzazione) e
// moltiplichiamo il risultato per SCALE per riportarlo alla scala
// originale (denormalizzazione).
const double SCALE = 100.0;

// ---------------------------------------------------------------------
// fileExists(filename)
// ---------------------------------------------------------------------
// Controlla se un file esiste ed è apribile, provando semplicemente ad
// aprirlo in lettura: se l'apertura fallisce (file assente, permessi
// mancanti, ecc.), std::ifstream si porta in uno stato di errore e
// f.good() restituisce false.
// Viene usata per decidere se caricare pesi già salvati in precedenza
// oppure se allenare la rete da zero.
bool fileExists(const std::string& filename)
{
    std::ifstream f(filename);
    return f.good();
}

int main()
{
    // Inizializziamo il generatore di numeri casuali con il tempo
    // corrente (std::time(0)), in modo che ogni esecuzione del
    // programma generi esempi di addestramento diversi. Senza questa
    // riga, rand() produrrebbe sempre la stessa sequenza di numeri
    // "casuali" a ogni avvio.
    std::srand(std::time(0));

    // Creiamo la rete: 2 neuroni di input (i due numeri da sommare),
    // 4 neuroni nello strato nascosto, 1 neurone di output (la somma).
    NeuralNetwork nn(2, 4, 1);

    // Nome del file in cui vengono salvati/caricati i pesi già allenati.
    std::string weightFile = "weights.txt";

    if (fileExists(weightFile))
    {
        // Se il file esiste, la rete è già stata allenata in una
        // esecuzione precedente: carichiamo i pesi invece di
        // riallenarla da zero (risparmiando tempo).
        std::cout << "Pesi trovati! Caricamento in corso..." << std::endl;
        nn.loadWeights(weightFile);
    }
    else
    {
        // Altrimenti alleniamo la rete da zero.
        std::cout << "Pesi non trovati. Avvio addestramento..." << std::endl;

        int epoche = 20000; // numero di esempi casuali su cui allenare la rete
        for (int i = 0; i < epoche; i++)
        {
            // Generiamo due numeri casuali tra 0 e 49 come esempio di
            // addestramento.
            double a = (std::rand() % 50);
            double b = (std::rand() % 50);

            // Chiamiamo train() passando input e target NORMALIZZATI
            // (divisi per SCALE), perché la rete lavora sempre con
            // valori compresi tra 0 e 1 (vedi il commento su SCALE
            // sopra). Ogni chiamata a train() esegue un singolo passo
            // di backpropagation e corregge un po' i pesi.
            nn.train({a / SCALE, b / SCALE}, {(a + b) / SCALE});

            // Ogni 2000 iterazioni stampiamo un esempio di previsione,
            // per poter osservare "a occhio" come la rete migliora
            // (l'errore tra "Rete" e il valore reale dovrebbe ridursi
            // man mano che si avanza nelle iterazioni).
            if (i % 2000 == 0)
            {
                std::vector<double> res = nn.predict({a / SCALE, b / SCALE});
                std::cout << "Iterazione " << i << " | " << a << " + " << b
                          << " = " << (a + b)
                          << " | Rete: " << res[0] * SCALE << std::endl;
            }
        }

        std::cout << "Addestramento completato." << std::endl;

        // Salviamo i pesi appena allenati su file, così la prossima
        // esecuzione del programma potrà riutilizzarli direttamente.
        nn.saveWeights(weightFile);
        std::cout << "Pesi salvati." << std::endl;
    }

    // Test finale: chiediamo alla rete di sommare 20 e 30 (valore
    // atteso: 50). Anche qui dividiamo per SCALE prima di passare i
    // valori alla rete (normalizzazione).
    double testA = 20.0, testB = 30.0;
    std::vector<double> res = nn.predict({testA / SCALE, testB / SCALE});

    // Moltiplichiamo il risultato per SCALE per riportarlo alla scala
    // originale (denormalizzazione) e lo stampiamo.
    std::cout << "Risultato: " << res[0] * SCALE << std::endl;

    return 0;
}
