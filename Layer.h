#ifndef LAYER_H
#define LAYER_H

#include "Matrix.h"

// =====================================================================
// Classe Layer
// =====================================================================
// Rappresenta UNO strato (layer) di neuroni della rete. Una rete
// neurale è composta da più Layer collegati in sequenza: l'output di
// uno strato diventa l'input dello strato successivo.
//
// Ogni Layer possiede:
//  - una matrice di PESI (weights): un numero per ogni collegamento
//    tra un neurone di ingresso e un neurone di uscita;
//  - un vettore di BIAS (biases): un numero per ogni neurone di
//    uscita, che viene sempre sommato indipendentemente dagli input.
//
// Questi due insiemi di numeri (pesi e bias) sono i "parametri" che la
// rete impara durante l'addestramento (train()) modificando i quali la
// rete migliora le sue previsioni.
// =====================================================================
class Layer {
public:
    int inputNodes;  // Numero di neuroni in ingresso a questo strato
    int outputNodes; // Numero di neuroni in uscita da questo strato

    Matrix weights;  // Matrice dei pesi (dimensioni: outputNodes x inputNodes)
    Matrix biases;   // Vettore colonna dei bias (dimensioni: outputNodes x 1)

    // -----------------------------------------------------------------
    // Costruttore: riceve il numero di neuroni in ingresso (inNodes) e
    // in uscita (outNodes), crea le matrici weights e biases con le
    // dimensioni corrette e le riempie con valori casuali (vedi
    // Layer.cpp per i dettagli).
    // -----------------------------------------------------------------
    Layer(int inNodes, int outNodes);

    // -----------------------------------------------------------------
    // feedForward(input)
    // -----------------------------------------------------------------
    // Calcola l'output di questo strato a partire da una matrice di
    // input (colonna di "inputNodes" valori), applicando:
    //   1. Il prodotto Pesi * Input
    //   2. La somma dei Bias
    //   3. La funzione di attivazione (sigmoide)
    // Restituisce una matrice colonna di "outputNodes" valori.
    // Non modifica pesi o bias: è un calcolo "in sola lettura", usato
    // sia durante la previsione (predict) sia durante il calcolo
    // preliminare dell'addestramento (train).
    // -----------------------------------------------------------------
    Matrix feedForward(const Matrix& input);
};

#endif // LAYER_H
