#include <cudaq.h>

#include "SurfaceCodeQubit.hpp"

__qpu__ void teleport(){
  cudaq::qarray<3> q;
  // Initial state preparation
  x(q[0]);

  // Create Bell pair
  h(q[1]);
  x<cudaq::ctrl>(q[1], q[2]);

  x<cudaq::ctrl>(q[0], q[1]);
  h(q[0]);

  auto b0 = mz(q[0]);
  auto b1 = mz(q[1]);

  if (b1)
    x(q[2]);
  if (b0)
    z(q[2]);

  mz(q[2]);

  return;
}

// lq is the 7 data qubit logical qubit register
void steane_h(cudaq::qview<> lq) __qpu__ {
  h(lq[0]);
  h(lq[1]);
  h(lq[2]);
  h(lq[3]);
  h(lq[4]);
  h(lq[5]);
  h(lq[6]);
}

// lq is the 7 data qubit logical qubit register
// x(lq) is valid, but only need final 3 qubits
void steane_x(cudaq::qview<> lq) __qpu__ {
  x(lq[4]);
  x(lq[5]);
  x(lq[6]);
}

// lq is the 7 data qubit logical qubit register
// z(lq) is valid, but only need final 3 qubits
void steane_z(cudaq::qview<> lq) __qpu__ {
  z(lq[4]);
  z(lq[5]);
  z(lq[6]);
}

// lq is the 7 data qubit logical qubit register
void steane_s(cudaq::qview<> lq) __qpu__ {
  s<cudaq::adj>(lq[0]);
  s<cudaq::adj>(lq[1]);
  s<cudaq::adj>(lq[2]);
  s<cudaq::adj>(lq[3]);
  s<cudaq::adj>(lq[4]);
  s<cudaq::adj>(lq[5]);
  s<cudaq::adj>(lq[6]);
}

// lq is the 7 data qubit logical qubit register
void steane_cx(cudaq::qview<> l_ctrl, cudaq::qview<> l_target) __qpu__ {
  x<cudaq::ctrl>(l_ctrl[0], l_target[0]);
  x<cudaq::ctrl>(l_ctrl[1], l_target[1]);
  x<cudaq::ctrl>(l_ctrl[2], l_target[2]);
  x<cudaq::ctrl>(l_ctrl[3], l_target[3]);
  x<cudaq::ctrl>(l_ctrl[4], l_target[4]);
  x<cudaq::ctrl>(l_ctrl[5], l_target[5]);
  x<cudaq::ctrl>(l_ctrl[6], l_target[6]);
}

void steane_prep_logical_zero_flagged(cudaq::qview<> q, cudaq::qview<> ancilla) __qpu__ {
  h(q[0]);
  h(q[4]);
  h(q[6]);
  x<cudaq::ctrl>(q[0], q[1]);
  x<cudaq::ctrl>(q[4], q[5]);
  x<cudaq::ctrl>(q[6], q[3]);
  x<cudaq::ctrl>(q[6], q[5]);
  x<cudaq::ctrl>(q[4], q[2]);
  x<cudaq::ctrl>(q[0], q[3]);
  x<cudaq::ctrl>(q[4], q[1]);
  x<cudaq::ctrl>(q[3], q[2]);

  x<cudaq::ctrl>(q[1], ancilla[0]);
  x<cudaq::ctrl>(q[3], ancilla[0]);
  x<cudaq::ctrl>(q[5], ancilla[0]);
}

void steane_prep_logical_zero(cudaq::qview<> q) __qpu__ {
  h(q[0]);
  h(q[4]);
  h(q[6]);
  x<cudaq::ctrl>(q[0], q[1]);
  x<cudaq::ctrl>(q[4], q[5]);
  x<cudaq::ctrl>(q[6], q[3]);
  x<cudaq::ctrl>(q[6], q[5]);
  x<cudaq::ctrl>(q[4], q[2]);
  x<cudaq::ctrl>(q[0], q[3]);
  x<cudaq::ctrl>(q[4], q[1]);
  x<cudaq::ctrl>(q[3], q[2]);
}

struct run_steane{
  void operator()() __qpu__ {
    cudaq::qvector q(7);
    cudaq::qvector anc(1);

    steane_prep_logical_zero_flagged(q, anc);
    auto flag = mz(anc);
    auto data = mz(q);
  }
};


struct stabilizer_memory{
  void operator()(uint32_t num_data_qubits, uint32_t rounds,
      const std::vector<std::vector<size_t>> &x_stabs,
      const std::vector<std::vector<size_t>> &z_stabs) __qpu__ {
    uint32_t num_x_ancillas = x_stabs.size();
    uint32_t num_z_ancillas = z_stabs.size();
    cudaq::qvector data(num_data_qubits);
    cudaq::qvector amx(num_x_ancillas);
    cudaq::qvector amz(num_z_ancillas);

    /* steane_prep_logical_zero(data); */

    // Starting with fresh qubits here,
    // in general should reset
    // Now apply x_stabilizer circuit
    // d = 2 example
    // s[0] = {0, 2}
    // s[1] = {1, 3}
    // So need 4 CX's:
    // CX(ax0, d0), CX(ax0, d2)
    // CX(ax1, d1), CX(ax1, d3)
    h(amx);
    for(size_t xi = 0; xi < x_stabs.size(); ++xi){
      for(size_t xj = 0; xj < x_stabs[xi].size(); ++xj){
        // anc index given by stab index
        // data index must be read out
        size_t di = x_stabs[xi][xj];
        x<cudaq::ctrl>(amx[xi], data[di]);
      }
    }
    h(amx);

    // Now apply z_stabilizer circuit
    for(size_t zi = 0; zi < z_stabs.size(); ++zi){
      for(size_t zj = 0; zj < z_stabs[zi].size(); ++zj){
        size_t di = z_stabs[zi][zj];
        x<cudaq::ctrl>(data[di], amz[zi]);
      }
    }

    auto x_readout1 = mz(amx);
    auto z_readout1 = mz(amz);

    bool multiround = 1;
    if(multiround){
      // round 2
      h(amx);
      for(size_t xi = 0; xi < x_stabs.size(); ++xi){
        for(size_t xj = 0; xj < x_stabs[xi].size(); ++xj){
          // anc index given by stab index
          // data index must be read out
          size_t di = x_stabs[xi][xj];
          x<cudaq::ctrl>(amx[xi], data[di]);
        }
      }
      h(amx);

      // Now apply z_stabilizer circuit
      for(size_t zi = 0; zi < z_stabs.size(); ++zi){
        for(size_t zj = 0; zj < z_stabs[zi].size(); ++zj){
          size_t di = z_stabs[zi][zj];
          x<cudaq::ctrl>(data[di], amz[zi]);
        }
      }

      auto x_readout2 = mz(amx);
      auto z_readout2 = mz(amz);

      // round 3
      h(amx);
      for(size_t xi = 0; xi < x_stabs.size(); ++xi){
        for(size_t xj = 0; xj < x_stabs[xi].size(); ++xj){
          // anc index given by stab index
          // data index must be read out
          size_t di = x_stabs[xi][xj];
          x<cudaq::ctrl>(amx[xi], data[di]);
        }
      }
      h(amx);

      // Now apply z_stabilizer circuit
      for(size_t zi = 0; zi < z_stabs.size(); ++zi){
        for(size_t zj = 0; zj < z_stabs[zi].size(); ++zj){
          size_t di = z_stabs[zi][zj];
          x<cudaq::ctrl>(data[di], amz[zi]);
        }
      }

      auto x_readout3 = mz(amx);
      auto z_readout3 = mz(amz);
    }

    auto data_readout = mz(data);
  }
};

int main() {
  uint32_t distance = 3;
  uint32_t rounds = 1;

  SurfaceCodeQubit scq(distance);
  scq.print_qubit_coords();
  scq.print_qubit_indices();
  scq.print_grid_map();
  scq.print_stabilizers();
  printf("x stabs: %zu\n", scq.x_stabilizers.size());
  printf("z stabs: %zu\n", scq.z_stabilizers.size());

  int nShots = 1000;
  // Sample
  // Run Surface code
  auto sc_counts = cudaq::sample(nShots, stabilizer_memory{}, distance*distance, rounds, scq.x_stabilizers, scq.z_stabilizers);
  printf("Surface code results:\n");
  sc_counts.dump();

  // Stein code
  std::vector<std::vector<size_t>> steane_x_stabilizers =
  {{0, 1, 2, 3},
    {1, 2, 4, 5},
    {2, 3, 5, 6}};

  std::vector<std::vector<size_t>> steane_z_stabilizers =
  {{0, 1, 2, 3},
    {1, 2, 4, 5},
    {2, 3, 5, 6}};

  uint32_t n_data_qubits = 7;
  /* auto counts = cudaq::sample(nShots, run_steane{}); */
  /* printf("Steane code results:\n"); */
  /* auto steane_counts = cudaq::sample(nShots, stabilizer_memory{}, n_data_qubits, rounds, steane_x_stabilizers, steane_z_stabilizers); */
  /* steane_counts.dump(); */
}

