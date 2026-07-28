#ifndef REACTORCLEMCONFIG_H
#define REACTORCLEMCONFIG_H

// Fernando-Clem: number of LEM elements reacted as one batched CVODE system
// by ReactorCvode::reactClemLine (see ReactorCvode.H/.cpp). PelePhysics is a
// shared physics library and must not reach back into a host code's headers
// (unlike the original CLEM fork, which included
// "ClemParticles/clem_index_definition.H" from here) - so this is a
// self-contained default, overridable at build time with -DNUM_LEM=<n>.
//
// MUST match the host's LEM line length (PeleC/CLEM: clem::config::n_lem,
// CLEM_NLEM in Source/Clem/ClemIndex.H) before the two are coupled - the
// batched reactor's persistent CVODE vector is sized for exactly NUM_LEM
// elements at initClemReactor() time.
#ifndef NUM_LEM
#define NUM_LEM 12
#endif

#endif
