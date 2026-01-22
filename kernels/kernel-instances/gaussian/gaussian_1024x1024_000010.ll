; ModuleID = 'kernels/kernel-template/gaussian_static_1.cl'
source_filename = "kernels/kernel-template/gaussian_static_1.cl"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: convergent nofree norecurse nounwind memory(argmem: readwrite) uwtable
define spir_kernel void @gaussian_1(ptr noalias nocapture noundef readonly align 4 %0, ptr noalias nocapture noundef readnone align 4 %1, ptr noalias nocapture noundef writeonly align 4 %2) local_unnamed_addr #0 !kernel_arg_addr_space !6 !kernel_arg_access_qual !7 !kernel_arg_type !8 !kernel_arg_base_type !8 !kernel_arg_type_qual !9 {
  %4 = tail call i64 @_Z12get_local_idj(i32 noundef 0) #3
  %5 = shl i64 %4, 6
  %6 = getelementptr i8, ptr %0, i64 4112
  %7 = getelementptr i8, ptr %0, i64 8224
  %8 = getelementptr i8, ptr %0, i64 12336
  %9 = getelementptr i8, ptr %0, i64 16448
  br label %10

10:                                               ; preds = %3, %25
  %11 = phi i1 [ true, %3 ], [ false, %25 ]
  %12 = phi i64 [ 0, %3 ], [ 512, %25 ]
  %13 = add i64 %12, %5
  br label %15

14:                                               ; preds = %25
  ret void

15:                                               ; preds = %10, %26
  %16 = phi i64 [ 0, %10 ], [ %27, %26 ]
  %17 = mul nuw nsw i64 %16, 1028
  %18 = getelementptr float, ptr %0, i64 %17
  %19 = getelementptr float, ptr %6, i64 %17
  %20 = getelementptr float, ptr %7, i64 %17
  %21 = getelementptr float, ptr %8, i64 %17
  %22 = getelementptr float, ptr %9, i64 %17
  %23 = shl nuw nsw i64 %16, 12
  %24 = getelementptr i8, ptr %2, i64 %23
  br label %29

25:                                               ; preds = %26
  br i1 %11, label %10, label %14

26:                                               ; preds = %29
  %27 = add nuw nsw i64 %16, 1
  %28 = icmp eq i64 %27, 1024
  br i1 %28, label %25, label %15

29:                                               ; preds = %15, %29
  %30 = phi i64 [ 0, %15 ], [ %112, %29 ]
  %31 = add nuw nsw i64 %30, %13
  %32 = getelementptr float, ptr %18, i64 %31
  %33 = load float, ptr %32, align 4, !tbaa !10
  %34 = add i64 %31, 1
  %35 = getelementptr float, ptr %18, i64 %34
  %36 = load float, ptr %35, align 4, !tbaa !10
  %37 = fmul float %36, 4.000000e+00
  %38 = tail call float @llvm.fmuladd.f32(float %33, float 2.000000e+00, float %37)
  %39 = add i64 %31, 2
  %40 = getelementptr float, ptr %18, i64 %39
  %41 = load float, ptr %40, align 4, !tbaa !10
  %42 = tail call float @llvm.fmuladd.f32(float %41, float 5.000000e+00, float %38)
  %43 = add i64 %31, 3
  %44 = getelementptr float, ptr %18, i64 %43
  %45 = load float, ptr %44, align 4, !tbaa !10
  %46 = tail call float @llvm.fmuladd.f32(float %45, float 4.000000e+00, float %42)
  %47 = add i64 %31, 4
  %48 = getelementptr float, ptr %18, i64 %47
  %49 = load float, ptr %48, align 4, !tbaa !10
  %50 = tail call float @llvm.fmuladd.f32(float %49, float 2.000000e+00, float %46)
  %51 = getelementptr float, ptr %19, i64 %31
  %52 = load float, ptr %51, align 4, !tbaa !10
  %53 = tail call float @llvm.fmuladd.f32(float %52, float 4.000000e+00, float %50)
  %54 = getelementptr float, ptr %19, i64 %34
  %55 = load float, ptr %54, align 4, !tbaa !10
  %56 = tail call float @llvm.fmuladd.f32(float %55, float 9.000000e+00, float %53)
  %57 = getelementptr float, ptr %19, i64 %39
  %58 = load float, ptr %57, align 4, !tbaa !10
  %59 = tail call float @llvm.fmuladd.f32(float %58, float 1.200000e+01, float %56)
  %60 = getelementptr float, ptr %19, i64 %43
  %61 = load float, ptr %60, align 4, !tbaa !10
  %62 = tail call float @llvm.fmuladd.f32(float %61, float 9.000000e+00, float %59)
  %63 = getelementptr float, ptr %19, i64 %47
  %64 = load float, ptr %63, align 4, !tbaa !10
  %65 = tail call float @llvm.fmuladd.f32(float %64, float 4.000000e+00, float %62)
  %66 = getelementptr float, ptr %20, i64 %31
  %67 = load float, ptr %66, align 4, !tbaa !10
  %68 = tail call float @llvm.fmuladd.f32(float %67, float 5.000000e+00, float %65)
  %69 = getelementptr float, ptr %20, i64 %34
  %70 = load float, ptr %69, align 4, !tbaa !10
  %71 = tail call float @llvm.fmuladd.f32(float %70, float 1.200000e+01, float %68)
  %72 = getelementptr float, ptr %20, i64 %39
  %73 = load float, ptr %72, align 4, !tbaa !10
  %74 = tail call float @llvm.fmuladd.f32(float %73, float 1.500000e+01, float %71)
  %75 = getelementptr float, ptr %20, i64 %43
  %76 = load float, ptr %75, align 4, !tbaa !10
  %77 = tail call float @llvm.fmuladd.f32(float %76, float 1.200000e+01, float %74)
  %78 = getelementptr float, ptr %20, i64 %47
  %79 = load float, ptr %78, align 4, !tbaa !10
  %80 = tail call float @llvm.fmuladd.f32(float %79, float 5.000000e+00, float %77)
  %81 = getelementptr float, ptr %21, i64 %31
  %82 = load float, ptr %81, align 4, !tbaa !10
  %83 = tail call float @llvm.fmuladd.f32(float %82, float 4.000000e+00, float %80)
  %84 = getelementptr float, ptr %21, i64 %34
  %85 = load float, ptr %84, align 4, !tbaa !10
  %86 = tail call float @llvm.fmuladd.f32(float %85, float 9.000000e+00, float %83)
  %87 = getelementptr float, ptr %21, i64 %39
  %88 = load float, ptr %87, align 4, !tbaa !10
  %89 = tail call float @llvm.fmuladd.f32(float %88, float 1.200000e+01, float %86)
  %90 = getelementptr float, ptr %21, i64 %43
  %91 = load float, ptr %90, align 4, !tbaa !10
  %92 = tail call float @llvm.fmuladd.f32(float %91, float 9.000000e+00, float %89)
  %93 = getelementptr float, ptr %21, i64 %47
  %94 = load float, ptr %93, align 4, !tbaa !10
  %95 = tail call float @llvm.fmuladd.f32(float %94, float 4.000000e+00, float %92)
  %96 = getelementptr float, ptr %22, i64 %31
  %97 = load float, ptr %96, align 4, !tbaa !10
  %98 = tail call float @llvm.fmuladd.f32(float %97, float 2.000000e+00, float %95)
  %99 = getelementptr float, ptr %22, i64 %34
  %100 = load float, ptr %99, align 4, !tbaa !10
  %101 = tail call float @llvm.fmuladd.f32(float %100, float 4.000000e+00, float %98)
  %102 = getelementptr float, ptr %22, i64 %39
  %103 = load float, ptr %102, align 4, !tbaa !10
  %104 = tail call float @llvm.fmuladd.f32(float %103, float 5.000000e+00, float %101)
  %105 = getelementptr float, ptr %22, i64 %43
  %106 = load float, ptr %105, align 4, !tbaa !10
  %107 = tail call float @llvm.fmuladd.f32(float %106, float 4.000000e+00, float %104)
  %108 = getelementptr float, ptr %22, i64 %47
  %109 = load float, ptr %108, align 4, !tbaa !10
  %110 = tail call float @llvm.fmuladd.f32(float %109, float 2.000000e+00, float %107)
  %111 = getelementptr float, ptr %24, i64 %31
  store float %110, ptr %111, align 4, !tbaa !10
  %112 = add nuw nsw i64 %30, 1
  %113 = icmp eq i64 %112, 64
  br i1 %113, label %26, label %29
}

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
