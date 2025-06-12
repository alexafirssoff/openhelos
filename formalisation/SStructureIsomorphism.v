Require Import Coq.Init.Datatypes.
Require Import Coq.Lists.List. Import ListNotations.
Require Import Coq.Logic.ProofIrrelevance.

Parameter I : Type.

Inductive S_list : Type :=
  | C1_list : I -> I -> S_list
  | C2_list : list (sum I S_list) -> I -> S_list
  | C3_list : S_list -> I -> S_list.

Definition RHS_list_type : Type :=
  sum (prod I I) (sum (prod (list (sum I S_list)) I) (prod S_list I)).

Definition slist_to_rhs (s_val : S_list) : RHS_list_type :=
  match s_val with
  | C1_list i1 i2    => inl (pair i1 i2)
  | C2_list l_is i   => inr (inl (pair l_is i))
  | C3_list s_rec i  => inr (inr (pair s_rec i))
  end.

Definition rhs_to_slist (rhs_val : RHS_list_type) : S_list :=
  match rhs_val with
  | inl (pair i1 i2)              => C1_list i1 i2
  | inr (inl (pair l_is i))       => C2_list l_is i
  | inr (inr (pair s_rec i))      => C3_list s_rec i
  end.

Lemma slist_to_rhs_circ_rhs_to_slist : forall (r : RHS_list_type), slist_to_rhs (rhs_to_slist r) = r.
Proof. intros r; destruct r as [ [i1 i2] | [ [l_is i_prod1] | [s_rec i_prod2] ] ]; simpl; reflexivity. Qed.

Lemma rhs_to_slist_circ_slist_to_rhs : forall (s_val : S_list), rhs_to_slist (slist_to_rhs s_val) = s_val.
Proof. intros s_val; destruct s_val; simpl; reflexivity. Qed.

Record iso (A B : Type) : Type := mk_iso {
  to      : A -> B;
  from    : B -> A;
  to_from : forall (b : B), to (from b) = b;
  from_to : forall (a : A), from (to a) = a
}.
Notation "A <~> B" := (iso A B) (at level 70).

Definition Slist_iso_RHS_v2 : S_list <~> RHS_list_type.
Proof.
  refine {| to      := slist_to_rhs;
            from    := rhs_to_slist;
            to_from := _;
            from_to := _
         |}.
  - (* forall b : RHS_list_type, slist_to_rhs (rhs_to_slist b) = b *)
    exact slist_to_rhs_circ_rhs_to_slist.
  - (* forall a : S_list, rhs_to_slist (slist_to_rhs a) = a *)
    exact rhs_to_slist_circ_slist_to_rhs.
Defined.


Print Slist_iso_RHS_v2.