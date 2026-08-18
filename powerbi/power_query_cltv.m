// =============================================
//  Power Query (M) — Import & Clean CLTV data
//  Use: Get Data > Blank Query > paste this
//  Table name after import: cltv_data
// =============================================

let
    // 1. Load the CSV exported by src/export_powerbi.py
    Source = Csv.Document(
        File.Contents("C:\path\to\powerbi_cltv_data.csv"),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]
    ),

    // 2. First row is the header
    PromoteHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),

    // 3. Set correct column types
    TypeColumns = Table.TransformColumnTypes(PromoteHeaders, {
        {"id", Int64.Type},
        {"gender", type text},
        {"area", type text},
        {"qualification", type text},
        {"income", type text},
        {"marital_status", Int64.Type},
        {"vintage", Int64.Type},
        {"claim_amount", type number},
        {"num_policies", type text},
        {"policy", type text},
        {"type_of_policy", type text},
        {"predicted_cltv", type number},
        {"cltv_segment", type text},
        {"income_band", type text},
        {"policy_band", type text},
        {"has_claims", Int64.Type}
    }),

    // 4. Friendly labels for readability
    AddGenderLabel = Table.AddColumn(TypeColumns, "gender_label",
        each if [gender] = "Male" then "Male" else "Female", type text),

    AddAreaLabel = Table.AddColumn(AddGenderLabel, "area_label",
        each if [area] = "Urban" then "Urban" else "Rural", type text),

    AddMaritalLabel = Table.AddColumn(AddAreaLabel, "marital_label",
        each if [marital_status] = 1 then "Married" else "Single", type text),

    // 5. Remove columns not needed for the dashboard
    RemoveColumns = Table.SelectColumns(AddMaritalLabel,
        {"id", "gender_label", "area_label", "qualification", "income_band",
         "marital_label", "vintage", "claim_amount", "policy_band", "policy",
         "type_of_policy", "predicted_cltv", "cltv_segment", "has_claims"})

in
    RemoveColumns
