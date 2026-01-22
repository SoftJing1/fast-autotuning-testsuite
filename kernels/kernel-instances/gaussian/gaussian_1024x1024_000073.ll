; ModuleID = 'kernels/kernel-template/gaussian_static_1.cl'
source_filename = "kernels/kernel-template/gaussian_static_1.cl"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: convergent nofree norecurse nounwind memory(argmem: readwrite) uwtable
define spir_kernel void @gaussian_1(ptr noalias nocapture noundef readonly align 4 %0, ptr noalias nocapture noundef readnone align 4 %1, ptr noalias nocapture noundef writeonly align 4 %2) local_unnamed_addr #0 !kernel_arg_addr_space !6 !kernel_arg_access_qual !7 !kernel_arg_type !8 !kernel_arg_base_type !8 !kernel_arg_type_qual !9 {
  %4 = tail call i64 @_Z12get_group_idj(i32 noundef 0) #3
  %5 = tail call i64 @_Z12get_local_idj(i32 noundef 0) #3
  %6 = shl i64 %4, 9
  %7 = shl i64 %5, 4
  %8 = add i64 %7, %6
  %9 = getelementptr i8, ptr %0, i64 4112
  %10 = getelementptr i8, ptr %0, i64 8224
  %11 = getelementptr i8, ptr %0, i64 12336
  %12 = getelementptr i8, ptr %0, i64 16448
  br label %13

13:                                               ; preds = %3, %28
  %14 = phi i64 [ 0, %3 ], [ %29, %28 ]
  %15 = shl nuw nsw i64 %14, 7
  %16 = add i64 %8, %15
  br label %18

17:                                               ; preds = %28
  ret void

18:                                               ; preds = %13, %31
  %19 = phi i64 [ 0, %13 ], [ %32, %31 ]
  %20 = mul nuw nsw i64 %19, 1028
  %21 = getelementptr float, ptr %0, i64 %20
  %22 = getelementptr float, ptr %9, i64 %20
  %23 = getelementptr float, ptr %10, i64 %20
  %24 = getelementptr float, ptr %11, i64 %20
  %25 = getelementptr float, ptr %12, i64 %20
  %26 = shl nuw nsw i64 %19, 12
  %27 = getelementptr i8, ptr %2, i64 %26
  br label %34

28:                                               ; preds = %31
  %29 = add nuw nsw i64 %14, 1
  %30 = icmp eq i64 %29, 4
  br i1 %30, label %17, label %13

31:                                               ; preds = %34
  %32 = add nuw nsw i64 %19, 1
  %33 = icmp eq i64 %32, 1024
  br i1 %33, label %28, label %18

34:                                               ; preds = %18, %34
  %35 = phi i64 [ 0, %18 ], [ %117, %34 ]
  %36 = add nuw nsw i64 %16, %35
  %37 = getelementptr float, ptr %21, i64 %36
  %38 = load float, ptr %37, align 4, !tbaa !10
  %39 = add i64 %36, 1
  %40 = getelementptr float, ptr %21, i64 %39
  %41 = load float, ptr %40, align 4, !tbaa !10
  %42 = fmul float %41, 4.000000e+00
  %43 = tail call float @llvm.fmuladd.f32(float %38, float 2.000000e+00, float %42)
  %44 = add i64 %36, 2
  %45 = getelementptr float, ptr %21, i64 %44
  %46 = load float, ptr %45, align 4, !tbaa !10
  %47 = tail call float @llvm.fmuladd.f32(float %46, float 5.000000e+00, float %43)
  %48 = add i64 %36, 3
  %49 = getelementptr float, ptr %21, i64 %48
  %50 = load float, ptr %49, align 4, !tbaa !10
  %51 = tail call float @llvm.fmuladd.f32(float %50, float 4.000000e+00, float %47)
  %52 = add i64 %36, 4
  %53 = getelementptr float, ptr %21, i64 %52
  %54 = load float, ptr %53, align 4, !tbaa !10
  %55 = tail call float @llvm.fmuladd.f32(float %54, float 2.000000e+00, float %51)
  %56 = getelementptr float, ptr %22, i64 %36
  %57 = load float, ptr %56, align 4, !tbaa !10
  %58 = tail call float @llvm.fmuladd.f32(float %57, float 4.000000e+00, float %55)
  %59 = getelementptr float, ptr %22, i64 %39
  %60 = load float, ptr %59, align 4, !tbaa !10
  %61 = tail call float @llvm.fmuladd.f32(float %60, float 9.000000e+00, float %58)
  %62 = getelementptr float, ptr %22, i64 %44
  %63 = load float, ptr %62, align 4, !tbaa !10
  %64 = tail call float @llvm.fmuladd.f32(float %63, float 1.200000e+01, float %61)
  %65 = getelementptr float, ptr %22, i64 %48
  %66 = load float, ptr %65, align 4, !tbaa !10
  %67 = tail call float @llvm.fmuladd.f32(float %66, float 9.000000e+00, float %64)
  %68 = getelementptr float, ptr %22, i64 %52
  %69 = load float, ptr %68, align 4, !tbaa !10
  %70 = tail call float @llvm.fmuladd.f32(float %69, float 4.000000e+00, float %67)
  %71 = getelementptr float, ptr %23, i64 %36
  %72 = load float, ptr %71, align 4, !tbaa !10
  %73 = tail call float @llvm.fmuladd.f32(float %72, float 5.000000e+00, float %70)
  %74 = getelementptr float, ptr %23, i64 %39
  %75 = load float, ptr %74, align 4, !tbaa !10
  %76 = tail call float @llvm.fmuladd.f32(float %75, float 1.200000e+01, float %73)
  %77 = getelementptr float, ptr %23, i64 %44
  %78 = load float, ptr %77, align 4, !tbaa !10
  %79 = tail call float @llvm.fmuladd.f32(float %78, float 1.500000e+01, float %76)
  %80 = getelementptr float, ptr %23, i64 %48
  %81 = load float, ptr %80, align 4, !tbaa !10
  %82 = tail call float @llvm.fmuladd.f32(float %81, float 1.200000e+01, float %79)
  %83 = getelementptr float, ptr %23, i64 %52
  %84 = load float, ptr %83, align 4, !tbaa !10
  %85 = tail call float @llvm.fmuladd.f32(float %84, float 5.000000e+00, float %82)
  %86 = getelementptr float, ptr %24, i64 %36
  %87 = load float, ptr %86, align 4, !tbaa !10
  %88 = tail call float @llvm.fmuladd.f32(float %87, float 4.000000e+00, float %85)
  %89 = getelementptr float, ptr %24, i64 %39
  %90 = load float, ptr %89, align 4, !tbaa !10
  %91 = tail call float @llvm.fmuladd.f32(float %90, float 9.000000e+00, float %88)
  %92 = getelementptr float, ptr %24, i64 %44
  %93 = load float, ptr %92, align 4, !tbaa !10
  %94 = tail call float @llvm.fmuladd.f32(float %93, float 1.200000e+01, float %91)
  %95 = getelementptr float, ptr %24, i64 %48
  %96 = load float, ptr %95, align 4, !tbaa !10
  %97 = tail call float @llvm.fmuladd.f32(float %96, float 9.000000e+00, float %94)
  %98 = getelementptr float, ptr %24, i64 %52
  %99 = load float, ptr %98, align 4, !tbaa !10
  %100 = tail call float @llvm.fmuladd.f32(float %99, float 4.000000e+00, float %97)
  %101 = getelementptr float, ptr %25, i64 %36
  %102 = load float, ptr %101, align 4, !tbaa !10
  %103 = tail call float @llvm.fmuladd.f32(float %102, float 2.000000e+00, float %100)
  %104 = getelementptr float, ptr %25, i64 %39
  %105 = load float, ptr %104, align 4, !tbaa !10
  %106 = tail call float @llvm.fmuladd.f32(float %105, float 4.000000e+00, float %103)
  %107 = getelementptr float, ptr %25, i64 %44
  %108 = load float, ptr %107, align 4, !tbaa !10
  %109 = tail call float @llvm.fmuladd.f32(float %108, float 5.000000e+00, float %106)
  %110 = getelementptr float, ptr %25, i64 %48
  %111 = load float, ptr %110, align 4, !tbaa !10
  %112 = tail call float @llvm.fmuladd.f32(float %111, float 4.000000e+00, float %109)
  %113 = getelementptr float, ptr %25, i64 %52
  %114 = load float, ptr %113, align 4, !tbaa !10
  %115 = tail call float @llvm.fmuladd.f32(float %114, float 2.000000e+00, float %112)
  %116 = getelementptr float, ptr %27, i64 %36
  store float %115, ptr %116, align 4, !tbaa !10
  %117 = add nuw nsw i64 %35, 1
  %118 = icmp eq i64 %117, 16
  br i1 %118, label %31, label %34
}

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare i64 @_Z12get_group_idj(i32 noundef) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare i64 @_Z12get_local_idj(i32 noundef) local_unnamed_addr #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #2

attributes #0 = { convergent nofree norecurse nounwind memory(argmem: readwrite) uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "uniform-work-group-size"="false" }
attributes #1 = { convergent mustprogress nofree nounwind willreturn memory(none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!opencl.ocl.version = !{!4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"uwtable", i32 2}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{i32 2, i32 0}
!5 = !{!"clang version 20.1.0"}
!6 = !{i32 1, i32 1, i32 1}
!7 = !{!"none", !"none", !"none"}
!8 = !{!"float*", !"float*", !"float*"}
!9 = !{!"restrict const", !"restrict", !"restrict"}
!10 = !{!11, !11, i64 0}
!11 = !{!"float", !12, i64 0}
!12 = !{!"omnipotent char", !13, i64 0}
!13 = !{!"Simple C/C++ TBAA"}
