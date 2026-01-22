; ModuleID = 'kernels/kernel-template/gaussian_static_1.cl'
source_filename = "kernels/kernel-template/gaussian_static_1.cl"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: convergent nofree norecurse nounwind memory(argmem: readwrite) uwtable
define spir_kernel void @gaussian_1(ptr noalias nocapture noundef readonly align 4 %0, ptr noalias nocapture noundef readnone align 4 %1, ptr noalias nocapture noundef writeonly align 4 %2) local_unnamed_addr #0 !kernel_arg_addr_space !6 !kernel_arg_access_qual !7 !kernel_arg_type !8 !kernel_arg_base_type !8 !kernel_arg_type_qual !9 {
  %4 = tail call i64 @_Z12get_local_idj(i32 noundef 0) #3
  %5 = shl i64 %4, 2
  %6 = getelementptr i8, ptr %0, i64 4112
  %7 = getelementptr i8, ptr %0, i64 8224
  %8 = getelementptr i8, ptr %0, i64 12336
  %9 = getelementptr i8, ptr %0, i64 16448
  br label %10

10:                                               ; preds = %3, %25
  %11 = phi i64 [ 0, %3 ], [ %26, %25 ]
  %12 = shl nuw nsw i64 %11, 4
  %13 = add i64 %12, %5
  br label %15

14:                                               ; preds = %25
  ret void

15:                                               ; preds = %10, %28
  %16 = phi i64 [ 0, %10 ], [ %29, %28 ]
  %17 = mul nuw nsw i64 %16, 1028
  %18 = getelementptr float, ptr %0, i64 %17
  %19 = getelementptr float, ptr %6, i64 %17
  %20 = getelementptr float, ptr %7, i64 %17
  %21 = getelementptr float, ptr %8, i64 %17
  %22 = getelementptr float, ptr %9, i64 %17
  %23 = shl nuw nsw i64 %16, 12
  %24 = getelementptr i8, ptr %2, i64 %23
  br label %31

25:                                               ; preds = %28
  %26 = add nuw nsw i64 %11, 1
  %27 = icmp eq i64 %26, 64
  br i1 %27, label %14, label %10

28:                                               ; preds = %31
  %29 = add nuw nsw i64 %16, 1
  %30 = icmp eq i64 %29, 1024
  br i1 %30, label %25, label %15

31:                                               ; preds = %15, %31
  %32 = phi i64 [ 0, %15 ], [ %114, %31 ]
  %33 = add nuw nsw i64 %32, %13
  %34 = getelementptr float, ptr %18, i64 %33
  %35 = load float, ptr %34, align 4, !tbaa !10
  %36 = add i64 %33, 1
  %37 = getelementptr float, ptr %18, i64 %36
  %38 = load float, ptr %37, align 4, !tbaa !10
  %39 = fmul float %38, 4.000000e+00
  %40 = tail call float @llvm.fmuladd.f32(float %35, float 2.000000e+00, float %39)
  %41 = add i64 %33, 2
  %42 = getelementptr float, ptr %18, i64 %41
  %43 = load float, ptr %42, align 4, !tbaa !10
  %44 = tail call float @llvm.fmuladd.f32(float %43, float 5.000000e+00, float %40)
  %45 = add i64 %33, 3
  %46 = getelementptr float, ptr %18, i64 %45
  %47 = load float, ptr %46, align 4, !tbaa !10
  %48 = tail call float @llvm.fmuladd.f32(float %47, float 4.000000e+00, float %44)
  %49 = add i64 %33, 4
  %50 = getelementptr float, ptr %18, i64 %49
  %51 = load float, ptr %50, align 4, !tbaa !10
  %52 = tail call float @llvm.fmuladd.f32(float %51, float 2.000000e+00, float %48)
  %53 = getelementptr float, ptr %19, i64 %33
  %54 = load float, ptr %53, align 4, !tbaa !10
  %55 = tail call float @llvm.fmuladd.f32(float %54, float 4.000000e+00, float %52)
  %56 = getelementptr float, ptr %19, i64 %36
  %57 = load float, ptr %56, align 4, !tbaa !10
  %58 = tail call float @llvm.fmuladd.f32(float %57, float 9.000000e+00, float %55)
  %59 = getelementptr float, ptr %19, i64 %41
  %60 = load float, ptr %59, align 4, !tbaa !10
  %61 = tail call float @llvm.fmuladd.f32(float %60, float 1.200000e+01, float %58)
  %62 = getelementptr float, ptr %19, i64 %45
  %63 = load float, ptr %62, align 4, !tbaa !10
  %64 = tail call float @llvm.fmuladd.f32(float %63, float 9.000000e+00, float %61)
  %65 = getelementptr float, ptr %19, i64 %49
  %66 = load float, ptr %65, align 4, !tbaa !10
  %67 = tail call float @llvm.fmuladd.f32(float %66, float 4.000000e+00, float %64)
  %68 = getelementptr float, ptr %20, i64 %33
  %69 = load float, ptr %68, align 4, !tbaa !10
  %70 = tail call float @llvm.fmuladd.f32(float %69, float 5.000000e+00, float %67)
  %71 = getelementptr float, ptr %20, i64 %36
  %72 = load float, ptr %71, align 4, !tbaa !10
  %73 = tail call float @llvm.fmuladd.f32(float %72, float 1.200000e+01, float %70)
  %74 = getelementptr float, ptr %20, i64 %41
  %75 = load float, ptr %74, align 4, !tbaa !10
  %76 = tail call float @llvm.fmuladd.f32(float %75, float 1.500000e+01, float %73)
  %77 = getelementptr float, ptr %20, i64 %45
  %78 = load float, ptr %77, align 4, !tbaa !10
  %79 = tail call float @llvm.fmuladd.f32(float %78, float 1.200000e+01, float %76)
  %80 = getelementptr float, ptr %20, i64 %49
  %81 = load float, ptr %80, align 4, !tbaa !10
  %82 = tail call float @llvm.fmuladd.f32(float %81, float 5.000000e+00, float %79)
  %83 = getelementptr float, ptr %21, i64 %33
  %84 = load float, ptr %83, align 4, !tbaa !10
  %85 = tail call float @llvm.fmuladd.f32(float %84, float 4.000000e+00, float %82)
  %86 = getelementptr float, ptr %21, i64 %36
  %87 = load float, ptr %86, align 4, !tbaa !10
  %88 = tail call float @llvm.fmuladd.f32(float %87, float 9.000000e+00, float %85)
  %89 = getelementptr float, ptr %21, i64 %41
  %90 = load float, ptr %89, align 4, !tbaa !10
  %91 = tail call float @llvm.fmuladd.f32(float %90, float 1.200000e+01, float %88)
  %92 = getelementptr float, ptr %21, i64 %45
  %93 = load float, ptr %92, align 4, !tbaa !10
  %94 = tail call float @llvm.fmuladd.f32(float %93, float 9.000000e+00, float %91)
  %95 = getelementptr float, ptr %21, i64 %49
  %96 = load float, ptr %95, align 4, !tbaa !10
  %97 = tail call float @llvm.fmuladd.f32(float %96, float 4.000000e+00, float %94)
  %98 = getelementptr float, ptr %22, i64 %33
  %99 = load float, ptr %98, align 4, !tbaa !10
  %100 = tail call float @llvm.fmuladd.f32(float %99, float 2.000000e+00, float %97)
  %101 = getelementptr float, ptr %22, i64 %36
  %102 = load float, ptr %101, align 4, !tbaa !10
  %103 = tail call float @llvm.fmuladd.f32(float %102, float 4.000000e+00, float %100)
  %104 = getelementptr float, ptr %22, i64 %41
  %105 = load float, ptr %104, align 4, !tbaa !10
  %106 = tail call float @llvm.fmuladd.f32(float %105, float 5.000000e+00, float %103)
  %107 = getelementptr float, ptr %22, i64 %45
  %108 = load float, ptr %107, align 4, !tbaa !10
  %109 = tail call float @llvm.fmuladd.f32(float %108, float 4.000000e+00, float %106)
  %110 = getelementptr float, ptr %22, i64 %49
  %111 = load float, ptr %110, align 4, !tbaa !10
  %112 = tail call float @llvm.fmuladd.f32(float %111, float 2.000000e+00, float %109)
  %113 = getelementptr float, ptr %24, i64 %33
  store float %112, ptr %113, align 4, !tbaa !10
  %114 = add nuw nsw i64 %32, 1
  %115 = icmp eq i64 %114, 4
  br i1 %115, label %28, label %31
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
