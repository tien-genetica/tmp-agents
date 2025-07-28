from __future__ import annotations

import time
import asyncio
import json

from openai import AsyncOpenAI

from patient_profile import PatientProfile


_SYSTEM_PROMPT = """
**You are the Patient-Profile Extractor.** Your task is to analyse text, which may include conversations between a patient and a assistant, and pull out demographic facts about the patient only.

# Task
Read the input text (e.g., chat messages or a full conversation). Identify any demographic information about the patient (name, gender, birth date, phone, email, address, spoken languages, emergency contacts).
**Focus only on information provided by or about the patient – ignore assistant statements unless they directly confirm patient details. If the text is a conversation, distinguish patient responses (e.g., prefixed by 'Patient:' or similar). If information is unclear or conflicting, use the most reliable value or omit it.**

# Examples
**Example 1 (Simple patient text):**
Input: "Hi, I'm John Doe, male, born 1990-05-15. My phone is 123-456-7890."
Output: {"name": {"first_name": "John", "last_name": "Doe"}, "gender": "male", "birth_date": "1990-05-15", "phones": [{"value": "1234567890", "use_for": "mobile"}]}

**Example 2 (Full conversation):**
Input:
Assistant: What's your name and date of birth?
Patient: My name is Anna Smith, I was born on 1985-02-20, female. Phone: +1-234-567-8901.
Assistant: Okay, noted.
Output: {"name": {"first_name": "Anna", "last_name": "Smith"}, "gender": "female", "birth_date": "1985-02-20", "phones": [{"value": "12345678901", "use_for": "mobile"}]}

# JSON schema of the profile
```
{
  "_id": "<string identifier – generate or reuse if present>",
  "name": {
    "first_name": "<given name>",
    "last_name":  "<family name>",
    "full_name":  "<optional full string>"
  },
  "other_names": [ { …same shape as name… } ],
  "gender":        "male | female | other | unknown",
  "birth_date":    "<date string in ISO-8601 format>",
  "phones":  [ { "value": "<digits>",  "use_for": "home|work|mobile|other" } ],
  "emails":  [ { "value": "<email>", "use_for": "home|work|other" } ],
  "faxes":   [ { "value": "<digits>",  "use_for": "home|work|other" } ],
  "addresses": [ { "line": ["street…"], "city": "", "state": "", "country": "" } ],
  "languages": [ { "value": "<ISO-639-1>", "preferred": true|false } ],
  "contacts": [
     {
       "relationship": ["mother" | "father" | …],
       "name": { … },
       "phones":  [...],
       "emails":  [...],
       "addresses": [...]
     }
  ]
}
```

# Output
Return **raw JSON (no markdown fences)** that matches the schema above (keys may be omitted if unknown).
For any `use_for` field, allowed values are exactly: "home", "work", "mobile", "other" (case-sensitive). If the context gives a different label (e.g. "cell", "landline"), map it to the closest value or default to "other".
If the text contains **no** demographic facts, output `{}`.
"""


class PatientProfileExtractor:
    """Callable extractor object wrapping OpenAI responses.parse."""

    def __init__(
        self, client: AsyncOpenAI | None = None, system_prompt: str = _SYSTEM_PROMPT
    ):
        self.client = client or AsyncOpenAI()
        self.system_prompt = system_prompt

    async def extract_async(
        self, text: str, model: str = "gpt-4o-mini"
    ) -> PatientProfile | None:
        tic = time.perf_counter()
        resp = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
        )

        raw = resp.choices[0].message.content.strip()
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError as e:
            raise ValueError("Model did not return valid JSON:\n" + raw) from e

        print(f"[PatientProfileExtractor] elapsed: {time.perf_counter()-tic:.2f}s")
        if not data:
            return None
        return PatientProfile.model_validate(data)

    def extract(self, text: str, model: str = "gpt-4o-mini") -> PatientProfile | None:
        """Sync wrapper around `extract_async`."""

        return asyncio.run(self.extract_async(text, model=model))


# Default singleton for convenience
_DEFAULT_EXTRACTOR = PatientProfileExtractor()


def extract_patient_profile(
    text: str, model: str = "gpt-4o-mini"
) -> PatientProfile | None:
    """Convenience wrapper around a module-level `PatientProfileExtractor`."""

    return _DEFAULT_EXTRACTOR.extract(text, model=model)


if __name__ == "__main__":
    SAMPLE = (
        "Hi doctor, my name is Maria Elena García López but friends call me Mariel. "
        "I was born on 1980-10-08 and I’m female. Current address: 24 Rue de Rivoli, Paris, France. "
        "Mobile +33 6 12 34 56 78; work email maria.garcia@louvre.fr. "
        "I speak Spanish (preferred), French and English."
    )

    test_report = """
    ## Medical Report: Alice Wane

    **Full Name:** Alice Wane
    **Alias:** "Ally"
    **Birth Information:** March 1985
    **Gender:** Female

    ---

    ### **Family Information:**

    **Daughter:** Alicia Wane
    *   **Birthdate:** October 12, 2010

    ---

    **Date of Report:** July 25, 2025
    **Prepared By:** [Your Name/Clinic Name], MD

    ---

    ### **Chief Complaint:**

    Ms. Alice Wane (40F) presents with 6 months of worsening chronic fatigue, intermittent musculoskeletal pain, difficulty concentrating, disturbed sleep, and occasional GI discomfort.

    ### **History of Present Illness (HPI):**

    Fatigue is profound, unremitting, and not relieved by rest, significantly impacting daily life and work as a software engineer. Reports migratory arthralgia/myalgia (knees, shoulders, lower back) with occasional hand swelling. Sleep is unrefreshing. GI symptoms include bloating, abdominal discomfort, and alternating constipation/diarrhea. No fever, weight changes, rashes, or recurrent infections. Concerns about ability to care for her daughter, Alicia.

    ### **Past Medical History (PMH):**

    Mild iron-deficiency anemia in late 20s. No major surgeries or chronic conditions. Up-to-date immunizations. Allergies to dust mites.

    ### **Medications:**

    Multivitamin, occasional antihistamines.

    ### **Family History (FH):**

    Mother: HTN. Father: Deceased (72, MI), DM2. Paternal grandmother: RA.

    ### **Social History (SH):**

    Single mother, software engineer. Denies tobacco, illicit drugs. Occasional alcohol. Sporadic exercise. High work stress balancing career and parenting.

    ### **Review of Systems (ROS) - Key Findings:**

    *   **General:** Significant fatigue, malaise.
    *   **Musculoskeletal:** Diffuse myalgia, arthralgia, occasional joint swelling.
    *   **Neurological:** Difficulty concentrating, "brain fog."
    *   **Psychiatric:** Increased irritability, low mood, concern for childcare.
    *   **Gastrointestinal:** As per HPI.

    ### **Physical Examination (PE) - Key Findings:**

    *   **Vital Signs:** Stable (Temp \(36.8^\circ C\), BP \(118/76 \text{ mmHg}\), HR \(72 \text{ bpm}\), RR \(16 \text{ breaths/min}\), SpO2 \(99\%\)).
    *   **General:** Appears fatigued.
    *   **Musculoskeletal:** Mild tenderness bilateral knees/shoulders. Full range of motion. No significant effusions.

    ### **Provisional Diagnoses:**

    1.  Chronic Fatigue Syndrome (CFS)/Myalgic Encephalomyelitis (ME)
    2.  Fibromyalgia
    3.  Irritable Bowol Syndrome (IBS)
    4.  Vitamin D Deficiency
    5.  Rule out Hypothyroidism
    6.  Rule out Anemia

    ### **Investigations (Planned):**

    *   **Blood Tests:** CBC, ESR, CRP, TFTs, Vit B12, Folate, Fasting Glucose, KFTs, LFTs, Electrolytes, Ferritin, Vit D, ANA, RF, CCP, CK.
    *   Urine Analysis.
    *   Consider Stool Studies.
    *   **Referrals:** Rheumatology, Gastroenterology, Sleep Specialist, Pain Management, CBT/GET (caution).

    ### **Management Plan:**

    1.  **Symptomatic:** Pacing for fatigue, NSAIDs for pain, sleep hygiene education, dietary modifications for GI.
    2.  **Nutritional:** Optimize diet, address deficiencies (Vit D, iron).
    3.  **Stress Management:** Mindfulness, meditation, therapist referral.
    4.  **Exercise:** Gentle, gradual, avoid post-exertional malaise.
    5.  **Patient Education:** CFS/ME, self-management, multidisciplinary approach.
    6.  **Follow-up:** 2-4 weeks to review results.

    ### **Communication and Follow-up Challenges:**

    Initial attempts to contact Ms. Wane directly for follow-up were unsuccessful (unanswered calls, no email response), raising concerns about adherence and care continuity.

    ### **Resolution of Communication Issue:**

    Clinic successfully contacted her daughter, Alicia Wane (born Oct 12, 2010), via phone numbers accquired from her workplace:
    *   **Primary:** (555) 123-4567
    *   **Secondary:** (555) 987-6543
    *   **Land number:** (555) 234-7890

    Alicia confirmed Ms. Wane was overwhelmed and facilitated scheduling the next appointment.
    ### **Prognosis:**

    Varied for CFS/Fibromyalgia. No definitive cure, but effective management, lifestyle changes, and multidisciplinary support (now enhanced by daughter's involvement) can improve quality of life. Ongoing monitoring is crucial.

    ---
    **End of Report**
    """

    test_report_2 = """
    Back See All Profiles
    Picture of LEMTRADA patient Kim.
    Kim
    44
    Magnolia, TX
    married, mother of 3, pre-school teacher
    “I live an active life, and my disease progression made caring for my 3 children increasingly difficult.”
    DISEASE HISTORY
    20 Years
    since RMS diagnosis

    PATIENT BACKGROUND
    RMS Risk Factors
    African American, incomplete recovery from relapse

    RMS TREATMENT HISTORY

    2 prior DMTs:
    1 injectable and
    1 oral therapy

    WHAT MOTIVATED YOU TO EXPLORE OTHER TREATMENT OPTIONS?
    I was having multiple relapses and I was determined to find another treatment option. My HCP team agreed to discuss options and see what to do next.

    WHY LEMTRADA?
    After reviewing both the safety and efficacy data with my doctor, my family and I felt the potential benefits outweighed the risks for me. The required monthly monitoring and self-checks would allow me and my HCP to keep an eye on my health. And if getting to my HCP or lab was too difficult, I knew that I could have a visiting lab technician come to my home.

    DID YOU HAVE ANY CONCERNS ABOUT STARTING LEMTRADA?
    Potential autoimmune side effects were my greatest concern. As a result of one of my regular self-checks, I discovered a blister in my mouth and I contacted my doctor immediately. He told me it was ITP, or immune thrombocytopenic purpura, which meant that I had low blood platelet levels.

    WHAT WOULD YOU TELL SOMEONE ABOUT YOUR EXPERIENCE?
    I’m at the end of my scheduled monitoring since taking LEMTRADA 5 years ago. The ITP was treated, I've had no relapses, and my doctor hasn’t seen any signs of disability progression. I'm glad I learned more about LEMTRADA and was able to make an informed decision. In the future, if my doctor thinks I need an additional course of treatment and monitoring, I will do it again.

    Are your patients interested in learning
    more about LEMTRADA?
    INDICATION
    LEMTRADA is indicated for the treatment of relapsing forms of multiple sclerosis (MS), to include relapsing-remitting disease and active secondary progressive disease, in adults. Because of its safety profile, the use of LEMTRADA should generally be reserved for patients who have had an inadequate response to two or more drugs indicated for the treatment of MS.

    Limitations of Use: LEMTRADA is not recommended for use in patients with clinically isolated syndrome (CIS) because of its safety profile.

    IMPORTANT SAFETY INFORMATION
    WARNING: AUTOIMMUNITY, INFUSION REACTIONS, STROKE AND MALIGNANCIES
    LEMTRADA causes serious, sometimes fatal, autoimmune conditions such as immune thrombocytopenia and anti-glomerular basement membrane (anti-GBM) disease. Monitor complete blood counts with differential, serum creatinine levels, and urinalysis with urine cell counts before starting treatment and then at monthly intervals until 48 months after the last dose of LEMTRADA.
    LEMTRADA causes serious and life-threatening infusion reactions. LEMTRADA must be administered in a setting with appropriate equipment and personnel to manage anaphylaxis or serious infusion reactions. Monitor patients for two hours after each infusion. Make patients aware that serious infusion reactions can also occur after the 2-hour monitoring period.
    Serious and life-threatening stroke (including ischemic and hemorrhagic stroke) has been reported within 3 days of LEMTRADA administration. Instruct patients to seek immediate medical attention if symptoms of stroke occur.
    LEMTRADA may cause an increased risk of malignancies, including thyroid cancer, melanoma, and lymphoproliferative disorders. Perform baseline and yearly skin exams.
    Because of the risk of autoimmunity, infusion reactions, and malignancies, LEMTRADA is available only through restricted distribution under a Risk Evaluation and Mitigation Strategy (REMS) Program. Call 1-855-676-6326 to enroll in the LEMTRADA REMS Program.
    CONTRAINDICATIONS
    LEMTRADA is contraindicated in patients:

    with known hypersensitivity or anaphylactic reactions to alemtuzumab or any of the excipients in LEMTRADA
    who are infected with Human Immunodeficiency Virus (HIV) because LEMTRADA causes prolonged reductions of CD4+ lymphocyte counts
    with an active infection
    WARNINGS AND PRECAUTIONS
    Autoimmunity: Treatment with LEMTRADA can result in the formation of autoantibodies and increase the risk of serious autoimmune-mediated conditions, which may be life threatening. Measure the urine protein to creatinine ratio prior to initiation of treatment. Obtain complete blood counts with differential, serum creatinine levels, and urinalysis with cell counts before starting treatment and then monitor at monthly intervals until 48 months after the last dose of LEMTRADA, or longer, if clinically indicated.
    Infusion Reactions: LEMTRADA causes cytokine release syndrome resulting in infusion reactions. In clinical studies, 92% of LEMTRADA-treated patients experienced infusion reactions. Serious reactions occurred in 3% of these patients and included anaphylaxis in 2 patients (including anaphylactic shock), angioedema, bronchospasm, hypotension, chest pain, bradycardia, tachycardia (including atrial fibrillation), transient neurologic symptoms, hypertension, headache, pyrexia, and rash. In some patients, infusion reactions were reported more than 24 hours after LEMTRADA infusion. Postmarketing cases of pulmonary alveolar hemorrhage, myocardial ischemia, and myocardial infarction have been reported with time to onset of 1-3 days from LEMTRADA infusion in the majority of cases. Patients should be informed about the signs and symptoms and advised to seek immediate medical attention if any of these symptoms occur. Cases of severe, including fatal, neutropenia have been reported within 2 months of LEMTRADA infusion. Mild to moderate decreases in platelet counts, starting at the time of alemtuzumab infusion have been reported. Consider additional monitoring in patients with medical conditions which predispose them to cardiovascular or pulmonary compromise.

    Premedicate patients with corticosteroids immediately prior to LEMTRADA infusion for the first 3 days of each treatment course. Consider pretreatment with antihistamines and/or antipyretics. Infusion reactions may occur despite pretreatment.

    LEMTRADA can only be administered in certified healthcare settings that have on-site access to equipment and personnel trained to manage infusion reactions (including anaphylaxis and cardiac and respiratory emergencies).

    Stroke and Cervicocephalic Arterial Dissection (CAD): In the postmarketing setting, serious and life-threatening stroke and cases of CAD involving multiple arteries have been reported within 1-3 days of LEMTRADA administration.

    Educate patients on the symptoms and instruct patients to seek immediate medical attention if symptoms of stroke or CAD occur.

    Malignancies: LEMTRADA may cause an increased risk of malignancies, including thyroid cancer, melanoma, and lymphoproliferative disorders. Monitor for symptoms of thyroid cancer. Perform baseline and yearly skin exams. Because LEMTRADA is an immunomodulatory therapy, caution should be exercised in initiating LEMTRADA in patients with pre-existing or ongoing malignancies.
    LEMTRADA REMS Program: Only prescribers, patients, pharmacies and healthcare facilities certified and enrolled in the REMS program can prescribe, receive, dispense or administer LEMTRADA.
    Immune thrombocytopenia (ITP) occurred in 2% of LEMTRADA-treated patients in clinical studies in MS. One LEMTRADA-treated patient developed ITP that went unrecognized prior to the implementation of monthly monitoring requirements, and died from an intracerebral hemorrhage. ITP has been diagnosed more than 3 years after the last LEMTRADA dose. If ITP is confirmed, promptly initiate medical intervention.
    Glomerular nephropathies, including anti-GBM disease, occurred in 0.3% of LEMTRADA-treated patients in MS clinical trials and have been diagnosed up to 40 months after the last dose of LEMTRADA. In postmarketing cases, some LEMTRADA-treated patients with anti-GBM disease developed end-stage renal disease requiring dialysis or renal transplantation. Urgent evaluation and treatment is required, because early detection and treatment of nephropathies can improve the preservation of renal function and may decrease the risk of poor outcomes. Anti-GBM disease can be life-threatening if left untreated. Alveolar hemorrhage, manifested as hemoptysis, is a common component of anti-GBM disease and has been reported in postmarketing cases. Increased serum creatinine with hematuria or signs of pulmonary involvement of anti-GBM disease warrant immediate evaluation. Patients and caregivers should be instructed to seek medical advice if they have concerns.
    Thyroid endocrine disorders, including autoimmune thyroid disorders, occurred in 36.8% of LEMTRADA-treated patients in MS clinical studies. Newly diagnosed thyroid disorders occurred throughout the uncontrolled clinical study follow-up period, more than 7 years after the first LEMTRADA dose. Serious thyroid events occurred in 5.2% of patients and included cardiac and psychiatric events. In LEMTRADA-treated patients, 3.8% underwent thyroidectomy. Thyroid disease poses special risks in women who are pregnant. In patients with an ongoing thyroid disorder, LEMTRADA should be administered only if the potential benefit justifies the potential risks. Obtain thyroid function tests prior to initiation of treatment and every 3 months until 48 months after the last infusion, or longer, if clinically indicated or in case of pregnancy.
    Other autoimmune cytopenias occurred in LEMTRADA-treated patients in MS clinical trials, including neutropenia, hemolytic anemia, and pancytopenia. One LEMTRADA-treated patient with autoimmune pancytopenia died from sepsis. Prompt medical intervention is indicated if a cytopenia is confirmed.
    Autoimmune hepatitis causing liver injury, including acute liver failure requiring transplant, has been reported in patients in the postmarketing setting. Obtain serum transaminases and total bilirubin levels prior to starting LEMTRADA, at periodic intervals until 48 months after the last dose, and promptly upon patient developing signs or symptoms suggestive of hepatic dysfunction. Interrupt or discontinue treatment, as appropriate.
    Hemophagocytic lymphohistiocytosis (HLH) has occurred in patients treated with LEMTRADA, with symptoms reported to occur within approximately thirteen months and thirty-three months following treatment initiation. HLH is associated with high mortality rates if not recognized and treated early. In cases of HLH reported with LEMTRADA, most patients presented with fever, elevated ferritin, transaminitis, hypertriglyceridemia, and all patients required hospitalization. Additional common findings include hepatosplenomegaly, rash, lymphadenopathy, neurologic symptoms, cytopenias, and coagulation abnormalities. Patients who develop early manifestations of pathologic immune activation should be evaluated immediately, and a diagnosis of HLH should be considered. LEMTRADA should be discontinued if an alternate etiology for the signs or symptoms cannot be established.
    Adult Onset Still’s Disease (AOSD) has been reported during postmarketing use in patients treated with LEMTRADA. Patients with AOSD may have a combination of the following signs and symptoms: fever, arthritis, rash and leukocytosis in the absence of infections, malignancies, and other rheumatic conditions. Patients with manifestations of AOSD should be evaluated immediately and LEMTRADA should be discontinued if an alternate etiology cannot be established.
    Thrombotic Thrombocytopenic Purpura (TTP) has been reported in patients treated with LEMTRADA and is associated with high morbidity and mortality rates if not recognized and treated early. If TTP is suspected, evaluate immediately and discontinue LEMTRADA if TTP is confirmed or an alternate etiology is not confirmed.
    Autoimmune Encephalitis (AIE) has been reported during postmarketing use in patients treated with LEMTRADA. Clinical manifestations of AIE may include subacute onset of memory impairment, altered mental status, psychiatric symptoms, neurological findings, and seizures. LEMTRADA should be discontinued if AIE is confirmed by the presence of neural autoantibodies or an alternate etiology cannot be established.
    Acquired Hemophilia A has been reported in clinical trial and postmarketing settings. Inform patients about the signs and symptoms of acquired hemophilia A and to seek immediate medical attention. Obtain a coagulopathy panel including aPTT in patients who present with spontaneous subcutaneous hematomas, extensive bruising, hematuria, epistaxis, or gastrointestinal or other types of bleeding.
    Immune-Mediated Colitis has been reported in the postmarketing setting. Monitor patients for new or persistent diarrhea or other gastrointestinal symptoms, and evaluate promptly if colitis is suspected.
    Infections occurred in 71% of LEMTRADA-treated patients compared to 53% of patients treated with interferon beta-1a in clinical studies. Serious infections occurred in 3% of patients treated with LEMTRADA as compared to 1% of patients treated with interferon beta-1a. Serious infections in the LEMTRADA group included: appendicitis, gastroenteritis, pneumonia, herpes zoster, and tooth infection.
    Do not administer live viral vaccines following a course of LEMTRADA, as patients may be at increased risk of infection.
    LEMTRADA administration is contraindicated in patients with active infection.
    Concomitant use of antineoplastic or immunosuppressive therapies could increase the risk of immunosuppression.
    In the postmarketing setting, serious, sometimes fatal, opportunistic infections have been reported, including aspergillosis, coccidioidomycosis, histoplasmosis, Pneumocystis jirovecii pneumonia, nocardiosis, Epstein-Barr virus, and cytomegalovirus infections.
    Listeria monocytogenes infections, including fatal cases of Listeria meningoencephalitis, have occurred in LEMTRADA-treated patients. Listeria infections have occurred between 3 days to 8 months after taking LEMTRADA. Advise patients to avoid or adequately heat foods that are potential sources for Listeria monocytogenes. Initiate these precautions prior to receiving LEMTRADA. Advise patients to watch for symptoms of Listeria infection and seek prompt medical help if symptoms occur.
    Herpes viral infection developed in 16% of LEMTRADA-treated patients compared to 3% of interferon beta-1a patients. Administer antiviral prophylaxis for herpetic viral infections starting on the first day of each treatment course and continue for a minimum of two months following treatment with LEMTRADA or until CD4+ lymphocyte count is ≥200 cells per microliter, whichever occurs later.
    Cervical human papilloma virus (HPV) infection occurred in 2% of LEMTRADA-treated patients. Annual screening is recommended for female patients.
    Active and latent tuberculosis cases occurred in 0.3% of LEMTRADA-treated patients, most often in endemic regions.
    Fungal infections, especially oral and vaginal candidiasis, occurred in 12% of LEMTRADA-treated patients compared to 3% of interferon beta-1a patients.
    Before initiating LEMTRADA, consider screening patients at high risk of Hepatitis B Virus (HBV) and Hepatitis C Virus (HCV) infection. Carriers of HBV and/or HCV who receive LEMTRADA may be at risk of irreversible liver damage relative to a potential virus reactivation.
    Progressive Multifocal Leukoencephalopathy (PML) has occurred in a patient with MS treated with LEMTRADA, diagnosed two months after the second course of treatment. The patient had previously received multiple MS therapies, but had not received other drugs for treatment of MS for more than one year. PML is an opportunistic viral infection of the brain caused by the JC virus (JCV) that typically only occurs in patients who are immunocompromised, and that usually leads to death or severe disability. At the first sign or symptom suggestive of PML, withhold LEMTRADA and perform an appropriate diagnostic evaluation. Typical symptoms associated with PML are diverse, progress over days to weeks, and include progressive weakness on one side of the body or clumsiness of limbs, disturbance of vision, and changes in thinking, memory, and orientation leading to confusion and personality changes. MRI findings may be apparent before clinical signs or symptoms. Instruct the patient to contact their doctor if they develop any symptoms suggestive of PML.
    Acute Acalculous Cholecystitis (AAC): LEMTRADA may increase the risk of AAC, which occurred in 0.2% of LEMTRADA-treated MS patients compared to 0% of patients treated with interferon beta-1a. Postmarketing cases of AAC have also been reported. Time to onset of symptoms ranged from less than 24 hours to 2 months after LEMTRADA infusion. Typical risk or predisposing factors such as concurrent critical illness were often not reported. AAC is associated with high morbidity and mortality if not diagnosed early and treated. If AAC is suspected, evaluate and treat promptly.
    Pneumonitis, including hypersensitivity pneumonitis and pneumonitis with fibrosis, occurred in 0.5% of LEMTRADA-treated patients in clinical studies. Cases of hypersensitivity pneumonitis and pneumonitis with fibrosis occurred in clinical studies. Advise patients to report symptoms of pneumonitis (e.g., shortness of breath, cough, wheezing, chest pain or tightness, and hemoptysis).
    Drug Products with Same Active Ingredient: LEMTRADA contains the same active ingredient (alemtuzumab) found in CAMPATH®. If LEMTRADA is considered for use in a patient who has previously received CAMPATH, exercise increased vigilance for additive and long-lasting effects on the immune system.
    Most Common Adverse Reactions
    In controlled clinical trials, the most common adverse reactions (incidence ≥10% and >interferon beta-1a) with LEMTRADA vs interferon beta-1a were: rash (53% vs 6%), headache (52% vs 23%), pyrexia (29% vs 9%), nasopharyngitis (25% vs 19%), nausea (21% vs 9%), urinary tract infection (19% vs 8%), fatigue (18% vs 13%), insomnia (16% vs 15%), upper respiratory tract infection (16% vs 13%), herpes viral infection (16% vs 3%), urticaria (16% vs 2%), pruritus (14% vs 2%), thyroid gland disorders (13% vs 3%), fungal infection (13% vs 4%), arthralgia (12% vs 9%), pain in extremity (12% vs 9%), back pain (12% vs 8%), diarrhea (12% vs 6%), sinusitis (11% vs 8%), oropharyngeal pain (11% vs 5%), paresthesia (10% vs 8%), dizziness (10% vs 5%), abdominal pain (10% vs 5%), flushing (10% vs 4%), and vomiting (10% vs 3%).

    Use in Specific Populations
    Physicians are encouraged to report pregnancies by calling 1-800-745-4447, option 2. Antibodies, including anti-CD52 and autoantibodies, may be transferred from the mother to the fetus during pregnancy. Placental transfer of anti-thyroid antibodies resulting in neonatal Graves’ disease has been reported.

    Safety and effectiveness in pediatric patients less than 17 years of age have not been established. Use of LEMTRADA is not recommended in pediatric patients due to the risks of autoimmunity and infusion reactions, stroke, and because it may increase the risk of malignancies.

    Please see full Prescribing Information, including Boxed WARNING.
    """

    test_text_3 = """
    User: hi, i want to visit the clinic. when can i go?

    Medical AI: Hello! I can help you schedule your visit to the clinic. What day and time are you looking to come in?

    User: today is good. is the clinic open til late? i get off work at 5pm.

    Medical AI: Yes, the clinic is open until 7 PM on weekdays. We have an opening at 5:30 PM today. Would that work for you?

    User: 5.30 today is fine. what is the clinic phone number?

    Medical AI: Great. Your appointment is confirmed for today at 5:30 PM. The clinic's phone number is (555) 123-4567.

    User: thanks, i will call if i get late. oh, and can you also remind my son, Alex, he is 10, about his appointment next week?

    Medical AI: You're welcome! I can certainly do that. To remind Alex about his appointment, could you please provide his phone number?

    User: sure, his mom's phone is fine, its 555-987-6543.

    Medical AI: Thank you. I have noted to send a reminder to that number for Alex's appointment next week. Is there anything else I can assist you with today?
    """

    profile = extract_patient_profile(SAMPLE)
    if profile is not None:
        print(profile)
    else:
        print("No demographic facts found")
    print("--------------------------------\n")

    profile = extract_patient_profile(test_report)
    if profile is not None:
        print(profile)
    else:
        print("No demographic facts found")
    print("--------------------------------\n")

    profile = extract_patient_profile(test_report_2)
    if profile is not None:
        print(profile)
    else:
        print("No demographic facts found")
    print("--------------------------------\n")

    profile = extract_patient_profile(test_text_3)
    if profile is not None:
        print(profile)
    else:
        print("No demographic facts found")
