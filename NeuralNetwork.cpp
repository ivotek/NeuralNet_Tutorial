#include "NeuralNetwork.h"
#include <iostream>

// ---------------------------------------------------------------------
// Costruttore NeuralNetwork::NeuralNetwork
// ---------------------------------------------------------------------
// Crea i due strati della rete:
//  - hiddenLayer: da "inputNodes" a "hiddenNodes" neuroni
//  - outputLayer: da "hiddenNodes" a "outputNodes" neuroni
// Il costruttore di ogni Layer inizializza già pesi e bias con valori
// casuali (vedi Layer.cpp), quindi qui basta impostare il tasso di
// apprendimento di default.
NeuralNetwork::NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes)
    : hiddenLayer(inputNodes, hiddenNodes),
      outputLayer(hiddenNodes, outputNodes)
{
    learningRate = 0.1; // Valore di default: quanto "velocemente" la rete corregge i pesi ad ogni passo
}

// ---------------------------------------------------------------------
// predict(input_array)  ->  calcola l'output della rete (senza allenarla)
// ---------------------------------------------------------------------
std::vector<double> NeuralNetwork::predict(const std::vector<double>& input_array)
{
    // 1. Converte l'array C++ (std::vector) in una Matrice colonna,
    //    perché tutte le operazioni della rete lavorano con oggetti
    //    Matrix e non con vector "semplici".
    Matrix inputs(input_array.size(), 1);
    for (size_t i = 0; i < input_array.size(); i++)
    {
        inputs.data[i][0] = input_array[i];
    }

    // 2. Feedforward attraverso il layer nascosto: dagli input grezzi
    //    (i due numeri) si ottiene la rappresentazione intermedia.
    Matrix hidden_output = hiddenLayer.feedForward(inputs);

    // 3. Feedforward attraverso il layer di output: dalla
    //    rappresentazione intermedia si ottiene il risultato finale.
    Matrix final_output = outputLayer.feedForward(hidden_output);

    // 4. Riconverte la Matrice finale in un array standard C++
    //    (std::vector), più comodo da usare fuori da questa classe
    //    (ad es. in main.cpp).
    std::vector<double> result(final_output.rows);
    for (int i = 0; i < final_output.rows; i++)
    {
        result[i] = final_output.data[i][0];
    }

    return result;
}

// ---------------------------------------------------------------------
// train(input_array, target_array)  ->  un passo di backpropagation
// ---------------------------------------------------------------------
// La backpropagation è l'algoritmo che permette alla rete di
// "imparare": si calcola quanto la previsione attuale si discosta dal
// valore atteso (l'errore), e si aggiornano i pesi e i bias in modo da
// ridurre quell'errore, partendo dall'ultimo strato (output) e
// tornando indietro fino al primo (hidden) — da qui il nome
// "propagazione all'indietro" dell'errore.
void NeuralNetwork::train(const std::vector<double>& input_array, const std::vector<double>& target_array)
{
    // ---- 1. INPUT ----
    // Come in predict(): trasformiamo l'input in una matrice colonna.
    Matrix inputs(input_array.size(), 1);
    for (size_t i = 0; i < input_array.size(); i++)
        inputs.data[i][0] = input_array[i];

    // ---- 2. FEEDFORWARD ----
    // Calcoliamo la previsione attuale della rete. A differenza di
    // predict(), qui teniamo da parte anche il risultato intermedio
    // "hidden" (l'output dello strato nascosto), perché servirà più
    // avanti per calcolare l'aggiornamento dei pesi.
    Matrix hidden = hiddenLayer.feedForward(inputs);
    Matrix output = outputLayer.feedForward(hidden);

    // ---- 3. CALCOLO ERRORE OUTPUT ----
    // Convertiamo il target atteso in una matrice colonna e calcoliamo
    // l'errore come differenza tra ciò che la rete AVREBBE dovuto
    // produrre (targets) e ciò che ha effettivamente prodotto (output).
    Matrix targets(target_array.size(), 1);
    for (size_t i = 0; i < target_array.size(); i++)
        targets.data[i][0] = target_array[i];
    Matrix output_errors = Matrix::subtract(targets, output);

    // ---- 4. CALCOLO GRADIENTE OUTPUT ----
    // Il gradiente ci dice "in che direzione e di quanto" muovere i
    // pesi per ridurre l'errore. Si calcola come:
    //     gradiente = derivata_sigmoide(output) * errore * learningRate
    // La derivata della sigmoide (dsigmoid) ci dice quanto "sensibile"
    // è l'output rispetto a piccole variazioni dei pesi: se il neurone
    // è già saturo (vicino a 0 o 1), la derivata è piccola e la
    // correzione sarà minore.
    Matrix gradients = Matrix::dsigmoid(output);
    gradients.multiply(output_errors);   // moltiplicazione elemento per elemento (Hadamard)
    gradients.multiply(learningRate);    // scala il gradiente in base al tasso di apprendimento

    // ---- 5. AGGIORNAMENTO PESI OUTPUT ----
    // La variazione da applicare ai pesi (delta) si ottiene
    // moltiplicando il gradiente per la trasposta dell'output dello
    // strato nascosto: in questo modo ogni peso viene corretto in
    // proporzione a quanto il neurone di ingresso corrispondente era
    // "attivo".
    //     delta_pesi = gradiente * trasposta(hidden)
    Matrix hidden_T = hidden.transpose();
    Matrix weight_deltas = Matrix::multiply(gradients, hidden_T);
    outputLayer.weights.add(weight_deltas); // pesi += delta
    outputLayer.biases.add(gradients);      // i bias si aggiornano semplicemente con il gradiente

    // ---- 6. BACKPROPAGATION DELL'ERRORE AL LAYER NASCOSTO ----
    // Per correggere anche i pesi dello strato nascosto, dobbiamo
    // prima capire "quanta parte" dell'errore di output è dovuta a
    // ciascun neurone nascosto. Questo si ottiene "retro-propagando"
    // l'errore attraverso la trasposta dei pesi dello strato di
    // output (che descrivono quanto ogni neurone nascosto influenza
    // ogni neurone di output).
    Matrix who_T = outputLayer.weights.transpose();
    Matrix hidden_errors = Matrix::multiply(who_T, output_errors);

    // ---- 7. CALCOLO GRADIENTE HIDDEN ----
    // Stesso ragionamento del passo 4, ma applicato allo strato
    // nascosto e al suo errore appena calcolato.
    Matrix hidden_gradient = Matrix::dsigmoid(hidden);
    hidden_gradient.multiply(hidden_errors);
    hidden_gradient.multiply(learningRate);

    // ---- 8. AGGIORNAMENTO PESI HIDDEN ----
    // Stesso ragionamento del passo 5: il delta dei pesi dello strato
    // nascosto si ottiene moltiplicando il suo gradiente per la
    // trasposta dell'input originale della rete.
    Matrix inputs_T = inputs.transpose();
    Matrix hidden_weight_deltas = Matrix::multiply(hidden_gradient, inputs_T);
    hiddenLayer.weights.add(hidden_weight_deltas);
    hiddenLayer.biases.add(hidden_gradient);
}

// ---------------------------------------------------------------------
// saveWeights(filename)  ->  salva pesi e bias di entrambi gli strati
// ---------------------------------------------------------------------
// Scrive su file, in sequenza e separati da uno spazio, tutti i valori
// di: hiddenLayer.weights, hiddenLayer.biases, outputLayer.weights,
// outputLayer.biases. L'ordine è importante: loadWeights() dovrà
// rileggerli esattamente nello stesso ordine.
void NeuralNetwork::saveWeights(const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Errore: impossibile aprire " << filename << " per la scrittura." << std::endl;
        return;
    }

    for (int i = 0; i < hiddenLayer.weights.rows; i++)
        for (int j = 0; j < hiddenLayer.weights.cols; j++)
            file << hiddenLayer.weights.data[i][j] << " ";

    for (int i = 0; i < hiddenLayer.biases.rows; i++)
        for (int j = 0; j < hiddenLayer.biases.cols; j++)
            file << hiddenLayer.biases.data[i][j] << " ";

    for (int i = 0; i < outputLayer.weights.rows; i++)
        for (int j = 0; j < outputLayer.weights.cols; j++)
            file << outputLayer.weights.data[i][j] << " ";

    for (int i = 0; i < outputLayer.biases.rows; i++)
        for (int j = 0; j < outputLayer.biases.cols; j++)
            file << outputLayer.biases.data[i][j] << " ";

    file.close();
}

// ---------------------------------------------------------------------
// loadWeights(filename)  ->  ricarica pesi e bias salvati in precedenza
// ---------------------------------------------------------------------
// Legge dal file gli stessi valori scritti da saveWeights(), nello
// stesso identico ordine, e li assegna alle matrici corrispondenti.
// In questo modo si evita di dover riallenare la rete da zero ad ogni
// avvio del programma (vedi main.cpp: se "weights.txt" esiste già,
// viene caricato invece di rifare tutto l'addestramento).
void NeuralNetwork::loadWeights(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Errore: impossibile aprire " << filename << " per la lettura." << std::endl;
        return;
    }

    for (int i = 0; i < hiddenLayer.weights.rows; i++)
        for (int j = 0; j < hiddenLayer.weights.cols; j++)
            file >> hiddenLayer.weights.data[i][j];

    for (int i = 0; i < hiddenLayer.biases.rows; i++)
        for (int j = 0; j < hiddenLayer.biases.cols; j++)
            file >> hiddenLayer.biases.data[i][j];

    for (int i = 0; i < outputLayer.weights.rows; i++)
        for (int j = 0; j < outputLayer.weights.cols; j++)
            file >> outputLayer.weights.data[i][j];

    for (int i = 0; i < outputLayer.biases.rows; i++)
        for (int j = 0; j < outputLayer.biases.cols; j++)
            file >> outputLayer.biases.data[i][j];

    file.close();
}
