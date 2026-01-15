-- Supabase Schema --
-- Run this in the Supabase SQL Editor to create the necessary tables for BRICK OS integration.

-- 1. Monomers Table
create table public.monomers (
  id text primary key,
  mw double precision,
  stiffness_gpa double precision,
  density_g_cc double precision,
  classification text
);

-- 2. Ballistic Threats Table
create table public.ballistic_threats (
  id text primary key,
  mass_g double precision,
  velocity_mps double precision,
  energy_j double precision
);

-- 3. Enable RLS (Optional but recommended)
alter table public.monomers enable row level security;
alter table public.ballistic_threats enable row level security;

-- 4. Create Policies (Allow Public Read, Service Role Write)
create policy "Allow Public Read" on public.monomers for select using (true);
create policy "Allow Public Read" on public.ballistic_threats for select using (true);

create policy "Allow Service Role Full Access" on public.monomers using (true) with check (true);
create policy "Allow Service Role Full Access" on public.ballistic_threats using (true) with check (true);
