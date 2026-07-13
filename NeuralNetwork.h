#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Layer.h"
#include <vector>
#include <fstream>
#include <string>

// =====================================================================
// Classe NeuralNetwork
// =====================================================================
// Rappresenta l'intera rete neurale, composta da due strati (Layer):
//   - hiddenLayer: lo strato nascosto, che riceve gli input grezzi
//     (in questo progetto: i due numeri da sommare) e li trasforma in
//     una rappresentazione intermedia.
//   - outputLayer: lo strato di uscita, che trasforma l'output dello
//     strato nascosto nel risultato finale (in questo progetto: la
//     somma stimata dei due numeri).
//
// Questa è un'architettura "feedforward" molto semplice, con un solo
// strato nascosto: è la struttura minima di una rete neurale "vera"
// (con più di un semplice strato input->output) capace di imparare
// funzioni non lineari.
// =====================================================================
class NeuralNetwork
{
public:
    Layer hiddenLayer;
    Layer outputLayer;
    double learningRate; // Tasso di apprendimento per l'aggiornamento dei pesi

    // -----------------------------------------------------------------
    // Costruttore
    // -----------------------------------------------------------------
    // inputNodes  = quanti numeri riceve la rete in ingresso (qui: 2)
    // hiddenNodes = quanti neuroni ha lo strato nascosto (qui: 4)
    // outputNodes = quanti numeri produce la rete in uscita (qui: 1)
    NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes);

    // -----------------------------------------------------------------
    // predict(input_array)
    // -----------------------------------------------------------------
    // Calcola la previsione della rete per un dato input, SENZA
    // modificare pesi o bias. E' la fase di "utilizzo" della rete,
    // usata sia per testare quanto ha imparato, sia internamente da
    // train() per calcolare gli output intermedi necessari alla
    // backpropagation.
    // -----------------------------------------------------------------
    std::vector<double> predict(const std::vector<double>& input_array);

    // -----------------------------------------------------------------
    // train(input_array, target_array)
    // -----------------------------------------------------------------
    // Esegue UN passo di addestramento (backpropagation):
    //  1. calcola la previsione attuale (feedforward);
    //  2. calcola l'errore rispetto al valore atteso (target);
    //  3. calcola quanto ogni peso ha contribuito all'errore
    //     (gradiente);
    //  4. corregge i pesi e i bias di entrambi gli strati in direzione
    //     opposta all'errore, proporzionalmente al learningRate.
    // Va chiamata ripetutamente (una volta per ogni esempio di
    // addestramento) per far convergere la rete verso previsioni
    // sempre più accurate.
    // -----------------------------------------------------------------
    void train(const std::vector<double>& input_array, const std::vector<double>& target_array);

    // -----------------------------------------------------------------
    // saveWeights(filename) / loadWeights(filename)
    // -----------------------------------------------------------------
    // Salvano su file (e ricaricano da file) tutti i pesi e i bias di
    // entrambi gli strati, in modo da poter riutilizzare una rete già
    // addestrata senza doverla riallenare da zero ogni volta che il
    // programma viene eseguito.
    // -----------------------------------------------------------------
    void saveWeights(const std::string& filename);
    void loadWeights(const std::string& filename);
};

#endif // NEURALNETWORK_H
