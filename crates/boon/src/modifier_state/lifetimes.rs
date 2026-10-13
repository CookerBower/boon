//! End applications from observed state transitions or intrinsic-source deletions.
use std::collections::HashMap;

use boon_proto::proto::CModifierTableEntry;

use crate::hero_stats::StatCatalog;
use crate::{
    Context, ModifierChange, modifier_state::ModifierApplication, player_states::StateEvidenceKeys,
};

type ModifierId = (u32, Option<u32>);
type StateMask = Vec<(usize, u32)>;

#[derive(Clone, Debug)]
pub(crate) struct StateDefinition {
    pub ability: Option<u32>,
    pub mask: StateMask,
    pub intrinsic: bool,
}

#[derive(Clone, Debug, Default)]
pub(super) struct ModifierLifetimes {
    keys: Option<StateEvidenceKeys>,
    masks: HashMap<ModifierId, (StateMask, bool)>,
    definitions: HashMap<u32, Vec<StateDefinition>>,
    candidates: HashMap<u32, (u32, ModifierId)>,
    observed: HashMap<u32, ModifierApplication>,
    observed_abilities: HashMap<u32, (ModifierApplication, u32)>,
    evidence: HashMap<u32, Option<Vec<u32>>>,
}

impl ModifierLifetimes {
    pub(super) fn from_catalog(catalog: &StatCatalog) -> Self {
        Self {
            definitions: catalog.modifier_state_definitions(),
            ..Self::default()
        }
    }

    pub(super) fn rebuild(&mut self, ctx: &Context, state: &HashMap<u32, CModifierTableEntry>) {
        self.clear();
        self.keys = ctx
            .serializers()
            .get("CCitadelPlayerPawn")
            .and_then(StateEvidenceKeys::resolve);
        for entry in state.values() {
            self.track(entry);
        }
    }

    pub(super) fn clear(&mut self) {
        self.keys = None;
        self.candidates.clear();
        self.observed.clear();
        self.observed_abilities.clear();
        self.evidence.clear();
    }

    fn track(&mut self, entry: &CModifierTableEntry) {
        let (Some(serial), Some(parent), Some(id)) =
            (entry.serial_number, entry.parent, entry.modifier_subclass)
        else {
            return;
        };
        let id = (id, entry.ability_subclass);
        let mask = self.masks.entry(id).or_insert_with(|| {
            let Some(definitions) = self.definitions.get(&id.0) else {
                return (Vec::new(), false);
            };
            let mut matching = definitions.iter().filter(|source| {
                id.1.is_none_or(|ability| {
                    ability == 0 || source.ability.is_none() || source.ability == Some(ability)
                })
            });
            let Some(source) = matching.next() else {
                return (Vec::new(), false);
            };
            if matching.next().is_some() {
                return (Vec::new(), false);
            }
            (source.mask.clone(), source.intrinsic)
        });
        if mask.0.is_empty() && !mask.1 {
            self.candidates.remove(&serial);
            self.observed.remove(&serial);
            self.observed_abilities.remove(&serial);
        } else {
            self.candidates.insert(serial, (parent, id));
        }
    }

    pub(super) fn expired(
        &mut self,
        ctx: &Context,
        state: &HashMap<u32, CModifierTableEntry>,
        changes: &[ModifierChange],
    ) -> Vec<u32> {
        if self.keys.is_none() {
            self.keys = ctx
                .serializers()
                .get("CCitadelPlayerPawn")
                .and_then(StateEvidenceKeys::resolve);
        }
        let mut expired = Vec::new();
        for change in changes {
            if let Some(entry) = state.get(&change.serial) {
                self.track(entry);
            } else {
                self.candidates.remove(&change.serial);
                self.observed.remove(&change.serial);
                self.observed_abilities.remove(&change.serial);
            }
        }
        self.evidence.clear();
        self.candidates.retain(|&serial, &mut (parent, id)| {
            let Some(entry) = state.get(&serial) else {
                self.observed.remove(&serial);
                self.observed_abilities.remove(&serial);
                return false;
            };
            let application = ModifierApplication::from(entry);
            let (mask, intrinsic) = &self.masks[&id];
            if *intrinsic
                && observe_ability(
                    &mut self.observed_abilities,
                    serial,
                    application,
                    entry.ability,
                    ctx.entities(),
                )
            {
                self.observed.remove(&serial);
                expired.push(serial);
                return false;
            }
            let evidence = self.evidence.entry(parent).or_insert_with(|| {
                let pawn = ctx.entities().get_by_handle(parent)?;
                (pawn.class_name.as_ref() == "CCitadelPlayerPawn")
                    .then(|| self.keys.as_ref()?.read(pawn))
                    .flatten()
            });
            let Some(words) = evidence else { return true };
            let present = mask.iter().try_fold(false, |present, &(index, bits)| {
                Some(present || words.get(index)? & bits != 0)
            });
            if !mask.is_empty() && observe(&mut self.observed, serial, application, present) {
                expired.push(serial);
                false
            } else {
                true
            }
        });
        expired.sort_unstable();
        expired
    }
}

fn observe_ability(
    observed: &mut HashMap<u32, (ModifierApplication, u32)>,
    serial: u32,
    application: ModifierApplication,
    handle: Option<u32>,
    entities: &crate::EntityContainer,
) -> bool {
    if observed
        .get(&serial)
        .is_some_and(|(old, _)| *old != application)
    {
        observed.remove(&serial);
    }
    let Some(handle) = handle else { return false };
    if entities.get_by_handle(handle).is_some() {
        observed.insert(serial, (application, handle));
        return false;
    }
    // Absence can come from filtering or missing packets. Only a recorded
    // deletion of a previously observed full handle can end an intrinsic effect.
    let deleted = entities.entity_changes().iter().any(|change| {
        change.kind == crate::EntityChangeKind::Deleted
            && change.id().index == (handle & crate::ENTITY_HANDLE_INDEX_MASK) as i32
            && change.id().serial == handle >> 14
    });
    deleted && observed.remove(&serial) == Some((application, handle))
}

fn observe(
    observed: &mut HashMap<u32, ModifierApplication>,
    serial: u32,
    application: ModifierApplication,
    present: Option<bool>,
) -> bool {
    if observed.get(&serial).is_some_and(|old| *old != application) {
        observed.remove(&serial);
    }
    match present {
        Some(true) => {
            observed.insert(serial, application);
            false
        }
        // Treat the catalog's declared states as required for this application.
        // Expiry is inferred, not a removal message; overlapping state providers
        // can hide it. Require a present-to-absent transition because a new
        // application's first tick may precede its mask. Missing evidence cannot end it.
        Some(false) => observed.remove(&serial).is_some(),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_HANDLE: u32 = (9 << 14) | 2;

    fn application(time: f32) -> ModifierApplication {
        ModifierApplication::from(&CModifierTableEntry {
            last_applied_time: Some(time),
            ..Default::default()
        })
    }

    fn source(serial: u32) -> crate::Entity {
        crate::Entity::from_fields(2, serial, 1, "Ability", true, Default::default()).unwrap()
    }

    fn deleted_source(serial: u32) -> crate::EntityContainer {
        let mut entities = crate::EntityContainer::new();
        entities.insert(source(serial)).unwrap();
        entities.clear_entity_changes();
        entities.insert(source(10)).unwrap();
        entities
    }

    fn observed_source() -> HashMap<u32, (ModifierApplication, u32)> {
        HashMap::from([(1, (application(10.0), SOURCE_HANDLE))])
    }

    #[test]
    fn observing_a_source_records_its_full_handle_without_expiry() {
        let mut entities = crate::EntityContainer::new();
        entities.insert(source(9)).unwrap();
        let mut observed = HashMap::new();
        assert!(!observe_ability(
            &mut observed,
            1,
            application(10.0),
            Some(SOURCE_HANDLE),
            &entities
        ));
        assert_eq!(observed, observed_source());
    }

    #[test]
    fn missing_source_evidence_does_not_expire_an_application() {
        for handle in [None, Some(SOURCE_HANDLE)] {
            let mut observed = observed_source();
            assert!(!observe_ability(
                &mut observed,
                1,
                application(10.0),
                handle,
                &crate::EntityContainer::new()
            ));
            assert_eq!(observed, observed_source());
        }
    }

    #[test]
    fn deletion_of_another_source_serial_does_not_expire_an_application() {
        let mut observed = observed_source();
        assert!(!observe_ability(
            &mut observed,
            1,
            application(10.0),
            Some(SOURCE_HANDLE),
            &deleted_source(8)
        ));
    }

    #[test]
    fn matching_source_deletion_can_expire_an_observed_application_only_once() {
        let mut observed = observed_source();
        let entities = deleted_source(9);
        assert!(observe_ability(
            &mut observed,
            1,
            application(10.0),
            Some(SOURCE_HANDLE),
            &entities
        ));
        assert!(!observe_ability(
            &mut observed,
            1,
            application(10.0),
            Some(SOURCE_HANDLE),
            &entities
        ));
    }

    #[test]
    fn another_modifier_cannot_reuse_a_source_observation() {
        let mut observed = observed_source();
        assert!(!observe_ability(
            &mut observed,
            2,
            application(10.0),
            Some(SOURCE_HANDLE),
            &deleted_source(9)
        ));
    }

    #[test]
    fn reapplication_requires_a_new_source_observation() {
        let mut observed = observed_source();
        assert!(!observe_ability(
            &mut observed,
            1,
            application(20.0),
            Some(SOURCE_HANDLE),
            &deleted_source(9)
        ));
    }

    #[test]
    fn catalog_masks_require_known_states_and_unique_matching_owners() {
        use serde_json::json;
        let file = |name, records| {
            serde_json::to_vec(&json!({"catalog":name,"client_version":"test","source_commit":"test","records":records})).unwrap()
        };
        let empty = json!([]);
        let heroes = file("heroes", &empty);
        let abilities = file("abilities", &empty);
        let misc = file("misc", &empty);
        let mut modifiers: serde_json::Value = serde_json::from_slice(&file("modifiers", &json!([
            {"record_key":"/one","definition_path":"/one","modifier_id":6,"qualified_modifier_id":11,"ability_id":123,"definition":{"m_nEnabledStateMask":"MODIFIER_STATE_TEST"},"stat_changes":[]},
            {"record_key":"/two","definition_path":"/two","modifier_id":6,"qualified_modifier_id":12,"ability_id":456,"definition":{"m_nEnabledStateMask":"MODIFIER_STATE_UNKNOWN"},"stat_changes":[]}
        ]))).unwrap();
        modifiers["modifier_states"] = json!({"35":"MODIFIER_STATE_TEST"});
        let catalog = StatCatalog::from_json_bytes(
            &heroes,
            &abilities,
            &serde_json::to_vec(&modifiers).unwrap(),
            &misc,
        )
        .unwrap();
        let mut reader = ModifierLifetimes::from_catalog(&catalog);
        for (serial, id, ability) in [
            (1, 6, Some(123)),
            (2, 6, None),
            (3, 12, Some(456)),
            (4, 11, Some(999)),
        ] {
            reader.track(&CModifierTableEntry {
                serial_number: Some(serial),
                parent: Some(2),
                modifier_subclass: Some(id),
                ability_subclass: ability,
                ..Default::default()
            });
        }
        assert_eq!(reader.candidates.len(), 1);
        assert!(reader.candidates.contains_key(&1));
        assert_eq!(reader.masks[&(6, Some(123))].0, [(1, 8)]);
    }

    #[test]
    fn absent_states_cannot_expire_an_unobserved_application() {
        assert!(!observe(
            &mut HashMap::new(),
            1,
            application(10.0),
            Some(false)
        ));
    }

    #[test]
    fn present_states_record_the_application_without_expiry() {
        let mut observed = HashMap::new();
        assert!(!observe(&mut observed, 1, application(10.0), Some(true)));
        assert_eq!(observed.get(&1), Some(&application(10.0)));
    }

    #[test]
    fn missing_states_preserve_a_previous_observation() {
        let mut observed = HashMap::from([(1, application(10.0))]);
        assert!(!observe(&mut observed, 1, application(10.0), None));
        assert!(observed.contains_key(&1));
    }

    #[test]
    fn state_expiry_does_not_clear_another_modifier() {
        let mut observed = HashMap::from([(1, application(10.0)), (2, application(10.0))]);
        assert!(observe(&mut observed, 1, application(10.0), Some(false)));
        assert_eq!(observed, HashMap::from([(2, application(10.0))]));
    }

    #[test]
    fn reapplication_cannot_expire_from_old_state_evidence() {
        let mut observed = HashMap::from([(1, application(10.0))]);
        assert!(!observe(&mut observed, 1, application(20.0), Some(false)));
    }
}
