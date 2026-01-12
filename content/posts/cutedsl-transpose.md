+++
title = 'Cute DSL Transpose'
date = 2026-01-11T15:50:46Z
draft = false
categories = ['cutedsl', 'transpose']
# date = '2023-04-16T16:03:45+0100'
# title: The big old test page
# subtitle: Fancy Subtitle
meta = true
# math: true
# toc: true
# hideDate: true
hideReadTime = true
description = "If the description field is not empty, its contents will show in the home page instead of the first 140 characters of the post."

+++

## Introduction

[CuteDSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html) is a new library that was release by Nvidia that allows us to write Cute-like kernels directly in Python. A kernel in CuteDSL is a function that is wrapped in `@cute.kernel` and launched from a how function wrapped in `@cute.jit`.

Transpose is a memory-bound operation that lends itself as a great introduction to CuteDSL because it needs a bunch of optimizations to make it fast.

I'll be assuming basic familiarity with how GPUs work.

## Getting The Baseline

Before starting it's always a good idea to define a target for our kernel's bandwidth.

Because I'm using an H100 GPU the peak bandwidth is `3.35 TB/s`. But achieving 100% throughput is practically impossible, so we'll have to settle for a more realistic baseline. Using `torch.compile`{{< sidenote >}}Using `x.T.contiguous()` without `torch.compile`, results in a bandwidth of `~350 GB/s`!{{< /sidenote >}} we can achieve a bandwidth of `~2.950TB/s`. Our job will be to write a faster kernel.

## Naive Implementation

The simplest implementation of transpose is for every thread to read an element from the input tensor and write it to the output tensor.

```python
class NaiveTranspose:
    def __init__(self, /, *, shape, num_warps=4):
        self.M, self.N = shape
        self.num_warps = num_warps

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mX_T: cute.Tensor,
    ):
        threads_per_block = 32 * self.num_warps

        self.kernel(mX, mX_T).launch(
            grid=(self.M * self.N // threads_per_block, 1, 1),
            block=(threads_per_block, 1, 1),
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mX_T: cute.Tensor,
    ):
        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]
        bdim = cute.arch.block_dim()[0]

        thread_idx = bidx * bdim + tidx

        n = mX.shape[1]
        ni = thread_idx % n
        mi = thread_idx // n
        mX_T[ni, mi] = mX[mi, ni]
```

This kernel has a bandwidth of `~200 GB/s`, which is around 15 times slower than our desired kernel. We can do better.

## Adding Coalesced Memory Access

The problem with the above code is twofold.

First, it stores a single float32 into a global memory which results in the following PTX `st.global.f32`. However we can instead store 4 float32s into a thread with a single instruction `st.global.v4.f32`. Doing this decreases the number of thread blocks launched by 4.

### Coalesced Writes Not Reads

and this applies to both global memory reads and writes.

Writing to global memory is always slower than reading, because when reading we can rely on caches.

### TVLayout

In order to implement our desired access pattern for a thread or for a thread block, we'll have to use a *Thread-Value Layout*, which consists of two layouts.

A *Value Layout* determines the shape of the sub-tensor that a single thread accesses. For example a

A *Thread Layout* determines the shape of the sub-tensor that a single thread accesses. For example a

if we have a Value layout with `v_rows` and `v_cols`, and a Thread layout with `t_rows` and `t_cols`. The number of rows and columns that the tensor has, has to be divisible by `t_rows * v_rows` and `t_cols * t_cols` respectively. This limitation must to be taken into account when writing kernels in the real world.

### Question

for more information on TVLayouts, see the [CuteDSL documentation]().
```question
Which of the following layouts is the fastest?

1. `thr_layout = cute.make_ordered_layout((threads_per_block, 1), order=(1, 0))`
2. `val_layout = cute.make_ordered_layout((1, self.elem_per_thread), order=(1, 0))`
3. `thr_layout = cute.make_ordered_layout((1, threads_per_block), order=(1, 0))`
4. `val_layout = cute.make_ordered_layout((self.elem_per_thread, 1), order=(1, 0))`
```

### Implementation

Putting the above all together.

```python
class CoalescedTranspose:
    def __init__(self, /, *, shape, num_warps=4, elem_per_thread=4, t_cols=1, v_rows=1, v_cols=4):
        self.M, self.N = shape
        self.num_warps = num_warps
        self.t_cols = t_cols
        self.v_rows = v_rows
        self.v_cols= v_cols

    @cute.jit
    def __call__(self, mX: cute.Tensor, mX_T: cute.Tensor):
        threads_per_block = 32 * self.num_warps
        mX_T_colmajor_layout = cute.make_layout((self.M, self.N))
        mX_T = cute.make_tensor(mX_T.iterator, mX_T_colmajor_layout)

        thr_layout = cute.make_ordered_layout((threads_per_block // self.t_cols, self.t_cols), order=(1, 0))
        val_layout = cute.make_ordered_layout((self.v_rows, self.v_cols), order=(1, 0))


        tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

        gX = cute.zipped_divide(mX, tiler_mn)
        gX_T = cute.zipped_divide(mX_T, tiler_mn)

        self.kernel(gX, gX_T, tv_layout).launch(
            grid=(cute.size(gX, mode=[1]), 1, 1),
            block=(cute.size(tv_layout, mode=[0]), 1, 1),
        )

    @cute.kernel
    def kernel(
        self,
        gX: cute.Tensor,
        gX_T: cute.Tensor,
        tv_layout: cute.Layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        blk_coord = ((None, None), bidx)

        blkX = gX[blk_coord]
        blkX_T = gX_T[blk_coord]

        tidfrgX = cute.composition(blkX, tv_layout)
        tidfrgX_T = cute.composition(blkX_T, tv_layout)

        thr_coord = (tidx, None)

        valX = tidfrgX[thr_coord]
        valX_T = tidfrgX_T[thr_coord]

        valX_T[None] = valX[None].load()
```

After autotuning, we can reach a bandwidth of `~2900 GB/s`, which is only `2%` slower than our target!

## Making The Reads Coalesced As Well


The above
