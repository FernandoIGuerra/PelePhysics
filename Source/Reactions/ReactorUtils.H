#ifndef REACTORUTILS_H
#define REACTORUTILS_H

#include <utility>
#include <AMReX_Gpu.H>
#include "mechanism.H"
#include "PelePhysics.H"
#include "ReactorTypes.H"
#include "AMReX_Sundials.H"

#include <arkode/arkode_arkstep.h>
#include <arkode/arkode_erkstep.h>
#include <cvode/cvode.h>

#include <nvector/nvector_serial.h>

#ifdef AMREX_USE_HIP
#include <nvector/nvector_hip.h>
#endif

#ifdef AMREX_USE_SYCL
#include <nvector/nvector_sycl.h>
#endif

#ifdef AMREX_USE_CUDA
#include <nvector/nvector_cuda.h>
#endif

namespace pele::physics::reactions::utils {
int check_flag(void* flagvalue, const char* funcname, int opt);

struct YCOrder
{
  // default: [ [Y_0,...Y_n]_{cell 0}, ..., [Y_0,...Y_n]_{cell m}]
  using ordering_type = YCOrder;
};

struct CYOrder
{
  // [ [Y_{cell 0},...Y_m]_{species 0}, ..., [Y_{cell 0},...Y_m]_{species n}]
  using ordering_type = CYOrder;
};

template <typename T>
struct is_ordering_type
{
  static constexpr bool value = std::is_base_of_v<typename T::ordering_type, T>;
};

template <typename OrderType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE int
vec_index(const int specie_idx, const int icell, const int /*ncells*/)
{
  return icell * (NUM_SPECIES + 1) + specie_idx;
}

template <typename OrderType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE int
spec_index(const int specie_idx, const int icell, const int /*ncells*/)
{
  return icell * NUM_SPECIES + specie_idx;
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE int
vec_index<CYOrder>(const int specie_idx, const int icell, const int ncells)
{
  return specie_idx * ncells + icell;
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE int
spec_index<CYOrder>(const int specie_idx, const int icell, const int ncells)
{
  return vec_index<CYOrder>(specie_idx, icell, ncells);
}

template <typename OrderType>
void
set_sundials_solver_tols(
  // cppcheck-suppress constParameter
  sundials::Context& sunctx,
  void* sundials_mem,
  const int ncells,
  const amrex::Real relTol,
  const amrex::Real absTol,
  const amrex::GpuArray<amrex::Real, NUM_SPECIES + 1>& typ_vals,
  const std::string& solvername,
  const int verbose)
{
  int omp_thread = 0;
#ifdef AMREX_USE_OMP
  omp_thread = omp_get_thread_num();
#endif

  const int neq_tot = (NUM_SPECIES + 1) * ncells;

#if defined(AMREX_USE_CUDA)
  N_Vector atol = N_VNewWithMemHelp_Cuda(
    neq_tot, /*use_managed_mem=*/false,
    *amrex::sundials::The_SUNMemory_Helper(), sunctx);
  amrex::Real* ratol = N_VGetDeviceArrayPointer_Cuda(atol);
#elif defined(AMREX_USE_HIP)
  N_Vector atol = N_VNewWithMemHelp_Hip(
    neq_tot, /*use_managed_mem=*/false,
    *amrex::sundials::The_SUNMemory_Helper(), sunctx);
  amrex::Real* ratol = N_VGetDeviceArrayPointer_Hip(atol);
#elif defined(AMREX_USE_SYCL)
  N_Vector atol = N_VNewWithMemHelp_Sycl(
    neq_tot, /*use_managed_mem=*/false,
    *amrex::sundials::The_SUNMemory_Helper(),
    &amrex::Gpu::Device::streamQueue(), sunctx);
  amrex::Real* ratol = N_VGetDeviceArrayPointer_Sycl(atol);
#else
  N_Vector atol = N_VNew_Serial(neq_tot, sunctx);
  amrex::Real* ratol = N_VGetArrayPointer(atol);
#endif

  if (typ_vals[0] > 0.0) {
    // cppcheck-suppress knownConditionTrueFalse
    if ((verbose > 0) && (omp_thread == 0)) {
      amrex::Print() << " Setting " << solvername
                     << " tolerances with TypVals rtol = " << relTol
                     << " atolfact = " << absTol << " in PelePhysics \n";
    }

    amrex::ParallelFor(ncells, [=] AMREX_GPU_DEVICE(int icell) {
      for (int k = 0; k < NUM_SPECIES + 1; k++) {
        ratol[vec_index<OrderType>(k, icell, ncells)] = typ_vals[k] * absTol;
      }
    });
  } else {
    // cppcheck-suppress knownConditionTrueFalse
    if ((verbose > 0) && (omp_thread == 0)) {
      amrex::Print() << " Setting " << solvername
                     << " tolerances rtol = " << relTol << " atol = " << absTol
                     << " in PelePhysics \n";
    }
    amrex::ParallelFor(
      neq_tot, [=] AMREX_GPU_DEVICE(int i) { ratol[i] = absTol; });
  }

  // Call CVodeSVtolerances to specify the scalar relative tolerance
  // and vector absolute tolerances
  int flag;
  if (solvername == "cvode") {
    flag = CVodeSVtolerances(sundials_mem, relTol, atol);
  } else if ((solvername == "arkstep") || (solvername == "erkstep")) {
    flag = ARKodeSVtolerances(sundials_mem, relTol, atol);
  } else {
    amrex::Abort("setSundialsSolverTols not implemented for this solver type");
  }
  if (utils::check_flag(&flag, "SVtolerances", 1) != 0) {
    amrex::Abort("Problem in setSundialsSolverTols");
  }

  N_VDestroy(atol);
}

template <typename OrderType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
box_flatten(
  const int icell,
  const int i,
  const int j,
  const int k,
  const int ncells,
  const int reactor_type,
  const bool clean_init_massfrac,
  amrex::Array4<const amrex::Real> const& rhoY,
  amrex::Array4<const amrex::Real> const& frcExt,
  amrex::Array4<const amrex::Real> const& temperature,
  amrex::Array4<const amrex::Real> const& rhoE,
  amrex::Array4<const amrex::Real> const& frcEExt,
  amrex::Real* y_vect,
  amrex::Real* src_vect,
  amrex::Real* vect_energy,
  amrex::Real* src_vect_energy)
{
  amrex::Real mass_frac[NUM_SPECIES];
  amrex::Real rho = 0.0;
  amrex::Real rho_old = 0.0;
  if (clean_init_massfrac) {
    for (int n = 0; n < NUM_SPECIES; n++) {
      rho_old += rhoY(i, j, k, n);
    }
  }
  for (int n = 0; n < NUM_SPECIES; n++) {
    const int idx = vec_index<OrderType>(n, icell, ncells);
    const int idx2 = spec_index<OrderType>(n, icell, ncells);
    const amrex::Real rhoYn =
      (clean_init_massfrac)
        ? amrex::min<amrex::Real>(
            amrex::max<amrex::Real>(0.0, rhoY(i, j, k, n)), rho_old)
        : rhoY(i, j, k, n);
    y_vect[idx] = rhoYn;
    src_vect[idx2] = frcExt(i, j, k, n);
    rho += rhoYn;
    mass_frac[n] = rhoYn;
  }
  const amrex::Real rho_inv = 1.0 / rho;
  for (amrex::Real& Yn : mass_frac) {
    Yn *= rho_inv;
  }

  amrex::Real temp = temperature(i, j, k, 0);
  vect_energy[icell] = rhoE(i, j, k, 0);
  src_vect_energy[icell] = frcEExt(i, j, k, 0);

  amrex::Real Enrg_loc = vect_energy[icell] * rho_inv;
  auto eos = pele::physics::PhysicsType::eos();
  if (reactor_type == ReactorTypes::e_reactor_type) {
    eos.REY2T(rho, Enrg_loc, mass_frac, temp);
  } else if (reactor_type == ReactorTypes::h_reactor_type) {
    eos.RHY2T(rho, Enrg_loc, mass_frac, temp);
  } else {
    amrex::Abort("Wrong reactor type. Choose between 1 (e) or 2 (h).");
  }
  y_vect[vec_index<OrderType>(NUM_SPECIES, icell, ncells)] = temp;
}

template <typename OrderType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
box_unflatten(
  const int icell,
  const int i,
  const int j,
  const int k,
  const int ncells,
  const int reactor_type,
  const bool clean_init_massfrac,
  amrex::Array4<amrex::Real> const& rhoY,
  amrex::Array4<amrex::Real> const& temperature,
  amrex::Array4<amrex::Real> const& rhoE,
  amrex::Array4<amrex::Real> const& frcEExt,
  amrex::Array4<amrex::Real> const& FC_in,
  const amrex::Real* y_vect,
  const amrex::Real* vect_energy,
  long int const& FCunt,
  amrex::Real dt)
{
  // Get the old Ys & rho as well as the clipped version
  // that went into CVODE
  amrex::Real rhoY_oldclipped[NUM_SPECIES];
  amrex::Real rho_old = 0.0;
  amrex::Real rho_oldclipped = 0.0;
  if (clean_init_massfrac) {
    for (int n = 0; n < NUM_SPECIES; n++) {
      rho_old += rhoY(i, j, k, n);
    }
    for (int n = 0; n < NUM_SPECIES; n++) {
      rhoY_oldclipped[n] = amrex::min<amrex::Real>(
        amrex::max<amrex::Real>(0.0, rhoY(i, j, k, n)), rho_old);
      rho_oldclipped += rhoY_oldclipped[n];
    }
  }

  amrex::Real mass_frac[NUM_SPECIES];
  amrex::Real rho = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++) {
    // With clean_init_massfrac, new rhoY equal old one + CVODE increment
    const amrex::Real rhoYn =
      (clean_init_massfrac)
        ? rhoY(i, j, k, n) + (y_vect[vec_index<OrderType>(n, icell, ncells)] -
                              rhoY_oldclipped[n]) *
                               rho_old / rho_oldclipped
        : y_vect[vec_index<OrderType>(n, icell, ncells)];
    rhoY(i, j, k, n) = rhoYn;
    rho += rhoYn;
    mass_frac[n] = rhoYn;
  }
  const amrex::Real rho_inv = 1.0 / rho;
  for (amrex::Real& Yn : mass_frac) {
    Yn *= rho_inv;
  }

  amrex::Real temp = y_vect[vec_index<OrderType>(NUM_SPECIES, icell, ncells)];
  rhoE(i, j, k, 0) = vect_energy[icell] + dt * frcEExt(i, j, k, 0);

  amrex::Real Enrg_loc = rhoE(i, j, k, 0) * rho_inv;
  auto eos = pele::physics::PhysicsType::eos();
  if (reactor_type == ReactorTypes::e_reactor_type) {
    eos.REY2T(rho, Enrg_loc, mass_frac, temp);
  } else if (reactor_type == ReactorTypes::h_reactor_type) {
    eos.RHY2T(rho, Enrg_loc, mass_frac, temp);
  } else {
    amrex::Abort("Wrong reactor type. Choose between 1 (e) or 2 (h).");
  }
  temperature(i, j, k, 0) = temp;
  FC_in(i, j, k, 0) = static_cast<amrex::Real>(FCunt);
}

template <typename OrderType>
class FlattenOps
{
public:
  static_assert(
    is_ordering_type<OrderType>::value,
    "FlattenOps must have ordering type as its template argument");

  void flatten(
    const amrex::Box& box,
    const int ncells,
    const int reactor_type,
    const bool clean_init_massfrac,
    amrex::Array4<const amrex::Real> const& rhoY,
    amrex::Array4<const amrex::Real> const& frcExt,
    amrex::Array4<const amrex::Real> const& temperature,
    amrex::Array4<const amrex::Real> const& rhoE,
    amrex::Array4<const amrex::Real> const& frcEExt,
    amrex::Real* y_vect,
    amrex::Real* src_vect,
    amrex::Real* vect_energy,
    amrex::Real* src_vect_energy)
  {
    BL_PROFILE("Pele::flatten()");
    const auto len = amrex::length(box);
    const auto lo = amrex::lbound(box);
    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      const int icell =
        (k - lo.z) * len.x * len.y + (j - lo.y) * len.x + (i - lo.x);
      box_flatten<OrderType>(
        icell, i, j, k, ncells, reactor_type, clean_init_massfrac, rhoY, frcExt,
        temperature, rhoE, frcEExt, y_vect, src_vect, vect_energy,
        src_vect_energy);
    });
  }

  void unflatten(
    const amrex::Box& box,
    const int ncells,
    const int reactor_type,
    const bool clean_init_massfrac,
    amrex::Array4<amrex::Real> const& rhoY,
    amrex::Array4<amrex::Real> const& temperature,
    amrex::Array4<amrex::Real> const& rhoE,
    amrex::Array4<amrex::Real> const& frcEExt,
    amrex::Array4<amrex::Real> const& FC_in,
    amrex::Real* y_vect,
    amrex::Real* vect_energy,
    long int* FCunt,
    amrex::Real dt)
  {
    BL_PROFILE("Pele::unflatten()");
    const auto len = amrex::length(box);
    const auto lo = amrex::lbound(box);
    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      const int icell =
        (k - lo.z) * len.x * len.y + (j - lo.y) * len.x + (i - lo.x);
      box_unflatten<OrderType>(
        icell, i, j, k, ncells, reactor_type, clean_init_massfrac, rhoY,
        temperature, rhoE, frcEExt, FC_in, y_vect, vect_energy, FCunt[icell],
        dt);
    });
  }
};

//--------------------------- CLEM IMPLEMENTATION -------------------------
// ClemFlatten / ClemUnflatten: box_flatten / box_unflatten for ONE CLEM
// element, the only cell (icell = 0) of an ncells = 1 CVODE vector. The
// element comes in primitive form (rho, Y_k, T, specific energy: e for the e
// reactor, h for the h reactor); the reactor type is a template parameter,
// so no run-time branch. There is no external source: the line operators
// (diffusion, stirring) are split from the reaction.

// element -> CVODE vector: rho Y_k and T (from the EOS, the element T as the
// first guess); rho energy and zero sources into the user data
template <typename OrderType, int ReactorType>
AMREX_FORCE_INLINE void
ClemFlatten(
  const bool clean_init_massfrac,
  const amrex::Real rho,
  const amrex::Real* mass_frac_in,
  const amrex::Real temperature,
  const amrex::Real energy,
  amrex::Real* y_vect,
  amrex::Real* src_vect,
  amrex::Real* vect_energy,
  amrex::Real* src_vect_energy)
{
  static_assert(
    ReactorType == ReactorTypes::e_reactor_type ||
      ReactorType == ReactorTypes::h_reactor_type,
    "ClemFlatten: wrong reactor type, 1 (e) or 2 (h)");
  constexpr int icell = 0;
  constexpr int ncells = 1;

  amrex::Real mass_frac[NUM_SPECIES];
  amrex::Real rho_ode = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++) {
    const amrex::Real rhoYn =
      (clean_init_massfrac)
        ? amrex::min<amrex::Real>(
            amrex::max<amrex::Real>(0.0, rho * mass_frac_in[n]), rho)
        : rho * mass_frac_in[n];
    y_vect[vec_index<OrderType>(n, icell, ncells)] = rhoYn;
    src_vect[spec_index<OrderType>(n, icell, ncells)] = 0.0;
    rho_ode += rhoYn;
    mass_frac[n] = rhoYn;
  }
  const amrex::Real rho_inv = 1.0 / rho_ode;
  for (amrex::Real& Yn : mass_frac) {
    Yn *= rho_inv;
  }

  vect_energy[icell] = rho * energy;
  src_vect_energy[icell] = 0.0;

  amrex::Real temp = temperature;
  const amrex::Real Enrg_loc = vect_energy[icell] * rho_inv;
  auto eos = pele::physics::PhysicsType::eos();
  if constexpr (ReactorType == ReactorTypes::e_reactor_type) {
    eos.REY2T(rho_ode, Enrg_loc, mass_frac, temp);
  } else {
    eos.RHY2T(rho_ode, Enrg_loc, mass_frac, temp);
  }
  y_vect[vec_index<OrderType>(NUM_SPECIES, icell, ncells)] = temp;
}

// CVODE vector -> element: the new Y_k (normalised by the integrated rho)
// and T from the EOS at the held rho energy. mass_frac holds the OLD Y_k on
// entry: with clean_init_massfrac the element gets old + CVODE increment, as
// in box_unflatten
template <typename OrderType, int ReactorType>
AMREX_FORCE_INLINE void
ClemUnflatten(
  const bool clean_init_massfrac,
  const amrex::Real rho,
  const amrex::Real energy,
  const amrex::Real* y_vect,
  amrex::Real* mass_frac,
  amrex::Real& temperature)
{
  constexpr int icell = 0;
  constexpr int ncells = 1;

  amrex::Real rhoY_oldclipped[NUM_SPECIES];
  amrex::Real rho_oldclipped = 0.0;
  if (clean_init_massfrac) {
    for (int n = 0; n < NUM_SPECIES; n++) {
      rhoY_oldclipped[n] = amrex::min<amrex::Real>(
        amrex::max<amrex::Real>(0.0, rho * mass_frac[n]), rho);
      rho_oldclipped += rhoY_oldclipped[n];
    }
  }

  amrex::Real rhoY[NUM_SPECIES];
  amrex::Real rho_ode = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++) {
    const amrex::Real y_n = y_vect[vec_index<OrderType>(n, icell, ncells)];
    rhoY[n] =
      (clean_init_massfrac)
        ? rho * mass_frac[n] + (y_n - rhoY_oldclipped[n]) * rho / rho_oldclipped
        : y_n;
    rho_ode += rhoY[n];
  }
  const amrex::Real rho_inv = 1.0 / rho_ode;
  for (int n = 0; n < NUM_SPECIES; n++) {
    mass_frac[n] = rhoY[n] * rho_inv;
  }

  amrex::Real temp = y_vect[vec_index<OrderType>(NUM_SPECIES, icell, ncells)];
  const amrex::Real Enrg_loc = rho * energy * rho_inv;
  auto eos = pele::physics::PhysicsType::eos();
  if constexpr (ReactorType == ReactorTypes::e_reactor_type) {
    eos.REY2T(rho_ode, Enrg_loc, mass_frac, temp);
  } else {
    eos.RHY2T(rho_ode, Enrg_loc, mass_frac, temp);
  }
  temperature = temp;
}
//--------------------------- CLEM IMPLEMENTATION -------------------------

template <typename OrderType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
fKernelSpec(
  const int icell,
  const int ncells,
  const amrex::Real dt_save,
  const int reactor_type,
  const amrex::Real* yvec_d,
  amrex::Real* ydot_d, // NOLINT(readability-non-const-parameter)
  const amrex::Real* rhoe_init,
  const amrex::Real* rhoesrc_ext,
  const amrex::Real* rYs,
  const pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type>*
    eosparm = nullptr)
{
  auto eos = pele::physics::PhysicsType::eos(eosparm);
  amrex::Real rho_pt = 0.0, rho_pt_inv = 0.0;
  amrex::GpuArray<amrex::Real, NUM_SPECIES> massfrac = {0.0}, massdens = {0.0};
  for (int n = 0; n < NUM_SPECIES; n++) {
    massdens[n] = yvec_d[vec_index<OrderType>(n, icell, ncells)];
  }
  eos.RY2RRinvY(massdens.data(), rho_pt, rho_pt_inv, massfrac.data());

  const amrex::Real nrg_pt =
    (rhoe_init[icell] + rhoesrc_ext[icell] * dt_save) * rho_pt_inv;

  amrex::Real temp_pt =
    yvec_d[vec_index<OrderType>(NUM_SPECIES, icell, ncells)];

  amrex::Real Cv_pt = 1.0;
  amrex::GpuArray<amrex::Real, NUM_SPECIES> ei_pt = {0.0};
#ifndef USE_MANIFOLD_EOS
  if (reactor_type == ReactorTypes::e_reactor_type) {
    eos.REY2T(rho_pt, nrg_pt, massfrac.arr, temp_pt);
    eos.RTY2Ei(rho_pt, temp_pt, massfrac.arr, ei_pt.arr);
    eos.RTY2Cv(rho_pt, temp_pt, massfrac.arr, Cv_pt);
  } else if (reactor_type == ReactorTypes::h_reactor_type) {
    eos.RHY2T(rho_pt, nrg_pt, massfrac.arr, temp_pt);
    eos.RTY2Hi(rho_pt, temp_pt, massfrac.arr, ei_pt.arr);
    eos.RTY2Cp(rho_pt, temp_pt, massfrac.arr, Cv_pt);
  } else {
    amrex::Abort("Wrong reactor type. Choose between 1 (e) or 2 (h).");
  }
#endif

  amrex::GpuArray<amrex::Real, NUM_SPECIES> cdots_pt = {0.0};
  eos.RTY2WDOT(rho_pt, temp_pt, massfrac.arr, cdots_pt.arr);

  amrex::Real rhoesrc = rhoesrc_ext[icell];
  for (int n = 0; n < NUM_SPECIES; n++) {
    const amrex::Real cdot_rYs =
      cdots_pt[n] + rYs[spec_index<OrderType>(n, icell, ncells)];
    ydot_d[vec_index<OrderType>(n, icell, ncells)] = cdot_rYs;
    rhoesrc -= cdot_rYs * ei_pt[n];
  }
  ydot_d[vec_index<OrderType>(NUM_SPECIES, icell, ncells)] =
    rhoesrc * (rho_pt_inv / Cv_pt);
}

#ifdef AMREX_USE_GPU
N_Vector
setNVectorGPU(int nvsize, int atomic_reduction, amrex::gpuStream_t stream);
#endif
} // namespace pele::physics::reactions::utils
#endif
