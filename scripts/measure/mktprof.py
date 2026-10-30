"""A phase-timed copy of a source tree: mktprof.py SRC DST (prints per-phase us/frame at exit)"""
import os, shutil, sys
src, dst = sys.argv[1], sys.argv[2]
shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
shutil.copy(f"{src}/CMakeLists.txt", dst); shutil.copytree(f"{src}/src", f"{dst}/src")
p = f"{dst}/src/imfeat/core.cpp"; s = open(p).read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:60])
    s = s.replace(old, new)
rep("namespace nb = nanobind;", r'''namespace nb = nanobind;
#include <chrono>
#include <cstdio>
namespace tprof {
inline double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#include <vector>
#include <algorithm>
struct T { double t[16] = {}; int n = 0; std::vector<double> all[16]; const char* nm[16] = {"acc_job","after_acc","derive_job","serial_levels","summary_job","summary_write","global","hashes","total"};
  void frame() { for (int i = 0; i < 9; ++i) { all[i].push_back(t[i]); t[i] = 0; } ++n; }
  ~T() { for (int i = 0; i < 9; ++i) { auto v = all[i]; if (v.empty() || *std::max_element(v.begin(), v.end()) == 0) continue; std::sort(v.begin(), v.end()); fprintf(stderr, "%-14s min %8.1f  median %8.1f us/frame\n", nm[i], 1e6 * v[v.size() / 10], 1e6 * v[v.size() / 2]); } } };
inline T& g() { static T t; return t; }
}
#define TP(i, stmt) do { const double _t0 = tprof::now(); stmt; tprof::g().t[i] += tprof::now() - _t0; } while (0)
''')
rep("    run_job(JOB_ACC);\n", "    TP(0, run_job(JOB_ACC));\n    const double _ta = tprof::now();\n")
rep('''    else if (levels_.size() == 1)
      global_sums();
  }''', '''    else if (levels_.size() == 1)
      global_sums();
    tprof::g().t[1] += tprof::now() - _ta;
  }''')
rep('''    run_job(JOB_DERIVE);
    if (stream_ && levels_.size() > 1) // the levels the bands could not share, then the global
    {
      for (size_t lvl = first_serial_level(); lvl < levels_.size(); ++lvl)
      {
        const Level &L = levels_[lvl];
        roll_up_rows(lvl, 0, L.ny);
        derive_run(L, 0, (size_t)L.ny * L.nx, L.buf.data(), bands_.size() == 1);
        if (bands_.size() == 1) summary_write(L);
      }
      global_sums();
    }''', '''    TP(2, run_job(JOB_DERIVE));
    if (stream_ && levels_.size() > 1) // the levels the bands could not share, then the global
    {
      TP(3, for (size_t lvl = first_serial_level(); lvl < levels_.size(); ++lvl)
      {
        const Level &L = levels_[lvl];
        roll_up_rows(lvl, 0, L.ny);
        derive_run(L, 0, (size_t)L.ny * L.nx, L.buf.data(), bands_.size() == 1);
        if (bands_.size() == 1) summary_write(L);
      });
      TP(6, global_sums());
    }''')
rep('''      run_job(JOB_SUMMARY);
      for (const Level &L : levels_) summary_write(L);''', '''      TP(4, run_job(JOB_SUMMARY));
      TP(5, for (const Level &L : levels_) summary_write(L));''')
rep('''    accumulate(img, rs, cs, chs, thumb_out);
    derive();
    if (!parallel_summary()) derive_hashes(); // (else the summary job did)''', '''    const double _tt = tprof::now();
    accumulate(img, rs, cs, chs, thumb_out);
    derive();
    if (!parallel_summary()) TP(7, derive_hashes()); // (else the summary job did)
    tprof::g().t[8] += tprof::now() - _tt; tprof::g().frame();''')
open(p, "w").write(s); print("tprof written")
